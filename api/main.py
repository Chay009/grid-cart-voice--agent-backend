from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# Load environment variables
load_dotenv()

# Initialize Neo4j and Groq API credentials
neo4j_uri = os.getenv('NEO4J_URI')
neo4j_password = os.getenv('NEO4J_PASSWORD')
neo4j_username = os.getenv('NEO4J_USERNAME')
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Initialize the LLM
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

# Define the query detection prompt template
query_detection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant that analyzes customer queries for an e-commerce platform. Categorize each query into one of the following:
        "Product Information",
        "Personalized Recommendations",
        "Deal or Bundle Offer",
        "Customer Support",
        "Cancel Subscription",
        "General",
        "Shipping"
        Provide your answer in the format 'Category: [category name]'."""),
    ("human", "Analyze the following customer query and determine its category:\n\n{query}")
])

# Create the query detection chain
query_detection_chain = query_detection_prompt | llm | StrOutputParser()

# Initialize the embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Initialize HuggingFaceEmbeddings
hug_face_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Initialize Neo4jGraph
graph = Neo4jGraph(
    url=neo4j_uri, 
    username=neo4j_username,
    password=neo4j_password, 
    database="neo4j",
    sanitize=True
)

CYPHER_GENERATION_TEMPLATE = """
Task: Generate a Cypher query based on the given schema and question. Ensure privacy and accuracy.

Schema:
"""""""
{schema}

Instructions:
1. Use only nodes, relationships, and properties from the schema.
2. Protect user privacy: don't query sensitive data like passwords or full purchase history.
3. For popularity, use COUNT() instead of size().
4. "You" refers to the seller, "I" refers to the user/customer.
5. Use case-insensitive and fuzzy search for text properties.
6. Limit results to prevent overwhelming responses.
7. Always use OPTIONAL MATCH for potentially missing relationships.
8. Use parameters for variable inputs (e.g., $userId instead of hardcoded values).
9. Include appropriate WHERE clauses to filter out null or irrelevant results.
10. For text searches, use toLower() for case-insensitive matching.
11. Use apoc.text.fuzzyMatch for more flexible text matching when appropriate.

Question: {question}

Cypher query:
"""

RESPONSE_TEMPLATE = """
Based on the Cypher query results, here's a response:

{response}

If you need more specific information or have any other questions, please ask.
"""

cypher_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["question"]
)

response_prompt = PromptTemplate(
    template=RESPONSE_TEMPLATE,
    input_variables=["response"]
)

def cypher_search(query):
    try:
        cypher_chain = GraphCypherQAChain.from_llm(
            graph=graph, 
            llm=llm, 
            verbose=True, 
            validate_cypher=True,
            cypher_prompt=cypher_prompt
        )
        
        result = cypher_chain.invoke(query)
        
        if not result:
            raise ValueError("No result from Cypher Chain")
        
        # Generate a response using the result
        response = llm(response_prompt.format(response=result))
        return response
    
    except Exception as e:
        print(f"Cypher search error: {str(e)}")
        return f"An error occurred while searching: {str(e)}"

def search_similar_products(graph, query_embedding, top_k=2, category=None):
    query = """
    CALL db.index.vector.queryNodes(
        "product_embedding_index", 
        $top_k, 
        $query_embedding
    ) YIELD node AS product, score
    WHERE product.isAvailable = true
    AND ($category IS NULL OR product.category = $category)
    RETURN 
        product.productId,
        product.title,
        product.description,
        product.price,
        product.category,
        product.imageLink,
        product.brand,
        product.stock,
        score
    ORDER BY score DESC
    """
    
    params = {
        "query_embedding": query_embedding,
        "top_k": top_k,
        "category": category
    }
    
    try:
        result = graph.query(query, params=params)
        return result
    except Exception as e:
        print(f"Error querying graph: {str(e)}")
        return []

# Define a Pydantic model for the request body
class QueryInput(BaseModel):
    query: str

@app.post("/")
async def brain(input_data: QueryInput):
    print('Received request')
    query = input_data.query
    try:
        # Detect query intent
        detected_intent = query_detection_chain.invoke({"query": query})
        
        # Generate query embedding
        query_embedding = model.encode(query)
        
        # Search for similar products
        results = search_similar_products(graph, query_embedding, top_k=5)
        
        # Perform Cypher search
        cypher_response = cypher_search(query)
        
        return {
            "category": detected_intent.strip(),
            "similar_products": results,
            "cypher_response": cypher_response
        }

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Note: The initialize_vector_index() function has been removed as it should be called separately