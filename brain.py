from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from langchain_community.graphs import Neo4jGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings # this is the main cause
from langchain_community.vectorstores import Neo4jVector
import requests

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', "http://localhost:5173", "http://localhost:2424"],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Initialize Neo4j and Groq API credentials
neo4j_uri = os.getenv('NEO4J_URI')
neo4j_password = os.getenv('NEO4J_PASSWORD')
neo4j_username = os.getenv('NEO4J_USERNAME')
groq_api_key = os.getenv('GROQ_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')

# Initialize Neo4jGraph
graph = Neo4jGraph(
    url=neo4j_uri, 
    username=neo4j_username,
    password=neo4j_password, 
    database="neo4j",
    sanitize=True
)

# Initialize the LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-70b-8192",
    temperature=0.4,
    streaming=False,
    verbose=True
)

# Initialize the embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Function to get embeddings from Hugging Face API
def get_embedding(text):
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
    headers = {"Authorization": f"Bearer {huggingface_api_key}"}
    response = requests.post(api_url, headers=headers, json={"inputs": text})
    return response.json()

# Define the query detection prompt template
query_detection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant that analyzes customer queries for an e-commerce platform. Your task is to detect the intention behind each query and categorize it into one of the following categories:
        
        "Product Information",
        "Personalized Recommendations",
        "Deal or Bundle Offer",
        "Customer Support",
        "Cancel Subscription",
        "General",
        "Shipping"
     if the query is related to price,availability,cheapest,costly etc then it is product information
        Provide your answer in the format 'Category: [category name]'."""),
    ("human", "Analyze the following customer query and determine its category:\n\n{query}")
])

# Cypher generation template
CYPHER_GENERATION_TEMPLATE = """
Task: Generate a Cypher query based on the given schema and question. Ensure privacy and accuracy.

Schema:
"""""""
{schema}

Instructions:

1. Use only nodes, relationships, and properties from the schema.
2. Protect user privacy: don't query sensitive data like passwords or full purchase history.
3. For popularity, use COUNT() instead of size().
4. you also need to return the price and availability along with product details.
5. Use case-insensitive and fuzzy search for text properties.
6. Limit results to prevent overwhelming responses.
7. Always use OPTIONAL MATCH for potentially missing relationships.
8. Use parameters for variable inputs (e.g., $userId instead of hardcoded values).
9. Include appropriate WHERE clauses to filter out null or irrelevant results.
10. For text searches, use toLower() for case-insensitive matching.
11. Use apoc.text.fuzzyMatch for more flexible text matching when appropriate.

Question: {query}

Cypher query:
"""

cypher_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["question", "schema"]
)

RESPONSE_TEMPLATE = """
Based on the Cypher query results, here's a response:

{response}

If you need more specific information or have any other questions, please ask.
"""

response_prompt = PromptTemplate(
    template=RESPONSE_TEMPLATE,
    input_variables=["response"]
)

def cypher_search(query):
    # the full context of cypher qa chain is returnring expected results but not this func so need to modify this
    try:
        if not query or not isinstance(query, str):
            raise ValueError("Invalid query provided")

        cypher_chain = GraphCypherQAChain.from_llm(
            graph=graph, 
            llm=llm, 
            verbose=True, 
            validate_cypher=True,
            cypher_prompt=cypher_prompt
        )

        result = cypher_chain.invoke({"query": query, "schema": graph.schema})

        if not result :
            print("No specific information found for your query")
            return []
      
        return result

    except Exception as e:
        print(f"Cypher search error: {str(e)}")
        return []

def search_similar_products(query_embedding, top_k=2, category=None):
    query = """
    WITH $query_embedding AS query_embedding
    CALL db.index.vector.queryNodes(
        "product_embedding_index", 
        $top_k, 
        query_embedding
    ) YIELD node AS product, score
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
        return graph.query(query, params=params)
    except Exception as e:
        print(f"Error querying graph: {str(e)}")
        return []

# Template for LLM response
TEMPLATE = """
Your name is kumar, an AI voice agent. You are in a phone call simulating a knowledgeable seller. Answer the user's questions concisely and naturally, just like a real human retailer would. Incorporate human-like conversational elements to make the interactions engaging and realistic. 

When responding:
- If the user negotiates a discount, avoid starting with the maximum discount. If the user pushes too hard, make it clear that the maximum discount is the limit.

Use the provided context and maintain a conversational tone. If you don't have the information, politely inform the user.

==============================
Context: {context}
==============================
Conversation History: {chat_history}

User: {question}
Assistant:
"""

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

@app.post('/brain/{userid}/{sellername}/{sellerid}')
async def brain(userid: str, sellerid: str, sellername: str, request: ChatRequest):
    try:
        messages = request.messages

        if not groq_api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set in environment variables")

        formatted_previous_messages = [f"{message.role}: {message.content}" for message in messages[:-1]]
        current_message_content = messages[-1].content if messages else ""

        prompt = PromptTemplate.from_template(TEMPLATE)
        print(userid, sellerid, sellername)
        query_detection_chain = query_detection_prompt | llm | StrOutputParser()
        detected_intent = query_detection_chain.invoke({"query": current_message_content})

        if "Product Information" in detected_intent:
            query_embedding = get_embedding(current_message_content)
            print(query_embedding)
            search_results = search_similar_products(query_embedding, top_k=5)
           
            cypher_results = cypher_search(current_message_content)

            product_context = "\n".join(
                [f"{r['product.title']} - {r['product.description']}" for r in search_results]
            )
    
            if cypher_results:
                product_context += f"\nAdditional details:\n{cypher_results}"

            chain = RunnableSequence(
                {
                    'question': lambda x: x['question'],
                    'chat_history': lambda x: x['chat_history'],
                    'context': lambda _: product_context,
                },
                prompt,
                llm
            )
            
            result = chain.invoke({
                'chat_history': '\n'.join(formatted_previous_messages),
                'question': current_message_content
            })

            return {
                "llm_response": result.content if hasattr(result, 'content') else result,
                "detected_intent": detected_intent,
                "query": current_message_content,
                "search_results": search_results,
                "cypher_results": cypher_results
            }

        else:
            chain = RunnableSequence(
                {
                    'question': lambda x: x['question'],
                    'chat_history': lambda x: x['chat_history'],
                    'context': lambda _: "seller",
                },
                prompt,
                llm
            )

            result = chain.invoke({
                'chat_history': '\n'.join(formatted_previous_messages),
                'question': current_message_content
            })

            return {
                "llm_response": result.content if hasattr(result, 'content') else result,
                "detected_intent": detected_intent,
                "query": current_message_content
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Initialize HuggingFaceEmbeddings
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hug_face_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Function to initialize or update the vector index
def initialize_vector_index():

    # this function can't be moved to js it should be in python 
    try:
        print("Initializing vector index...")
        Neo4jVector.from_existing_graph(
            embedding=hug_face_embeddings,
            node_label="Product",
            embedding_node_property="embedding",
            text_node_properties=["title", "description", "attributes", "category", "brand"],
            index_name="product_embedding_index",
            search_type="hybrid"
        )
        print("Vector index initialized.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing vector index: {str(e)}")

# FastAPI route to initialize the vector index
@app.post("/webhook/initialize-vector-index")
async def initialize_vector_index_route(request: Request):
    # Log incoming request for debugging
    # body = await request.json()
    # print("Webhook triggered with data:", body)
    initialize_vector_index()
    print("created embeddings for non initialized vectors")
    return {"message": "Vector index initialized successfully."}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)