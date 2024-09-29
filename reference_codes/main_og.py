from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
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
    allow_origins=['http://localhost:3000'],  # client URL
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Initialize the LLM
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

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
        Provide your answer in the format 'Category: [category name]'."""),
    ("human", "Analyze the following customer query and determine its category:\n\n{query}")
])

# Create the query detection chain
query_detection_chain = query_detection_prompt | llm | StrOutputParser()
# Define the schema and template for generating Cypher queries
CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database strictly based on the schema and instructions provided.

Instructions:
1. Use only nodes, relationships, and properties mentioned in the schema.
2. Always enclose the Cypher output inside 3 backticks. Do not add 'cypher' after the backticks.
3. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Product title use `toLower(p.title) CONTAINS toLower('shoes')`
4. Always use aliases to refer the node in the query.
5. Use count(DISTINCT n) for aggregations to avoid duplicates.
6. Use OPTIONAL MATCH for potentially missing relationships to prevent query failures.
7. Include appropriate WHERE clauses to filter out null or irrelevant results.
8. Use parameters for variable inputs (e.g., $userId instead of hardcoded values).
9. Limit results to prevent overwhelming responses (use LIMIT clause).
10. For popularity or ranking queries, use ORDER BY with LIMIT.
11. Use WITH clauses to structure complex queries and improve readability.
12. Avoid accessing sensitive information like passwords or full purchase histories.
13. "You" refers to the seller, "I" or "me" refers to the user/customer in queries.
14. Use apoc.text.fuzzyMatch for more flexible text matching when appropriate.

Schema:
Product {productId: STRING, description: STRING, price: FLOAT, category: STRING, brand: STRING, stock: FLOAT, createdAt: STRING, updatedAt: STRING, isAvailable: BOOLEAN, imageLink: STRING, attributes: STRING, title: STRING, embedding: LIST}
User {userId: STRING, username: STRING, purchaseHistory: LIST}
Seller {sellerId: STRING, sellername: STRING, products: LIST}
Category {name: STRING}
Brand {name: STRING}

The relationships:
(:User)-[:PURCHASED {purchaseDate: STRING}]->(:Product)
(:User)-[:INTERESTED_IN]->(:Category)
(:User)-[:INTERESTED_IN]->(:Brand)
(:User)-[:PREFERS]->(:Product)
(:Seller)-[:SELLS]->(:Product)

# Example 1: Get product recommendations for user with userId "12345" based on interests and preferences
MATCH (u:User {userId: $userId})
OPTIONAL MATCH (u)-[:INTERESTED_IN]->(c:Category)<-[:INTERESTED_IN]-(similarUser:User)
OPTIONAL MATCH (similarUser)-[:PURCHASED]->(p:Product)
WHERE NOT (u)-[:PURCHASED]->(p) AND p.isAvailable = true
WITH DISTINCT p, count(similarUser) as similarity
ORDER BY similarity DESC
RETURN p.title, p.description, p.price
LIMIT 5

# Example 2: Get products sold by sellers of products that the user has purchased
MATCH (u:User {userId: $userId})
OPTIONAL MATCH (u)-[:PURCHASED]->(purchasedProduct:Product)<-[:SELLS]-(seller:Seller)
OPTIONAL MATCH (seller)-[:SELLS]->(otherProduct:Product)
WHERE NOT (u)-[:PURCHASED]->(otherProduct) AND otherProduct.isAvailable = true
WITH DISTINCT otherProduct, count(purchasedProduct) as relevance
ORDER BY relevance DESC
RETURN otherProduct.title, otherProduct.description, otherProduct.price
LIMIT 5

# Example 3: Get popular products in categories that the user is interested in
MATCH (u:User {userId: $userId})-[:INTERESTED_IN]->(c:Category)<-[:INTERESTED_IN]-(p:Product)<-[:SELLS]-(s:Seller)
WHERE p.isAvailable = true
WITH p, count(DISTINCT s) as sellerCount
ORDER BY sellerCount DESC
LIMIT 10
RETURN p.title, p.description, p.price, sellerCount as popularity

# Example 4: Find products with fuzzy title match
MATCH (p:Product)
WHERE p.isAvailable = true AND apoc.text.fuzzyMatch(p.title, $searchTerm)
RETURN p.title, p.description, p.price
ORDER BY apoc.text.levenshteinSimilarity(p.title, $searchTerm) DESC
LIMIT 5

The question is:
{question}
"""

# Initialize the embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Initialize HuggingFaceEmbeddings
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hug_face_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize Neo4jGraph
graph = Neo4jGraph(
    url=neo4j_uri, 
    username=neo4j_username,
    password=neo4j_password, 
    database="neo4j",  # Ensure this is correctly specified
    sanitize=True
)

# Function to initialize or update the vector index
# this func should be seperate route and whenever a new product is created call this to api so that
# it refreshes

def initialize_vector_index():
    print("Initializing vector index...")
    Neo4jVector.from_existing_graph(
        embedding=hug_face_embeddings,
        node_label="Product",
        embedding_node_property="embedding",
        text_node_properties=["title", "description", "attributes", "category", "brand"],
        index_name="product_embedding_index",  # Ensure this name matches
        search_type="hybrid"
    )
    print("Vector index initialized.")

# Call this function separately to initialize the vector index for coverting embeddings without creating one
#try to move it to server of crud admin/seller server
initialize_vector_index()


# this old has some points as comments also try testing prompting for accuracy
# def old_cyper_search(query):

    # here we need to pass two tmeplates one to instruct for better cypher query and the other is to respond
    # as of now when no result found it tell i dont know


# this test case gave this error so modify template to latest cypher queries
#{code: Neo.ClientError.Statement.SyntaxError} {message: A pattern expression should only be used in order to test the existence of a pattern. It can no longer be used inside the function size(), an alternative is to replace size() with COUNT {}. (line 2, column 14 (offset: 53))
# " for example WITH p, size((s)-[:SELLS]->(p)) as popularity"
# for this question--- recommend me the product which is popular in your store

# in our contxt "you" means the seller and 'i" mean user/customer it should be included in template too
# for example if we ask whom you sold the product it is telling seller names 
# when tried without providing it the schema it is not search attributes 
    # cypher_chain = GraphCypherQAChain.from_llm(graph=graph, 
    #                                            llm=llm, 
    #                                            verbose=True, validate_cypher=True)
    
    # #
    
    # res=cypher_chain.invoke(query)
    # print(res)
    # cypher_prompt = PromptTemplate(
    # template = cypher_generation_template,
    # input_variables = ["schema", "question"]
  # this error when this query products price whose price is aroung 40000"
 # Cypher search error: {code: Neo.ClientError.Statement.TypeError} {message: Expected a string value for `toLower`, but got: Double(2.700000e+02); consider converting it to a string with toString().}

# one important fix needed is atributes which should be fixed in product creation as text

# here "you" should be mapped to selllerId
# missing values are from schema in this try passing schema from input 
CYPHER_GENERATION_TEMPLATE = """
Task: Generate a Cypher query based on the given schema and question. Ensure privacy and accuracy.

Schema:
{schema}

Instructions:
1. Use only nodes, relationships, and properties from the schema.
2. Protect user privacy: don't query sensitive data like passwords or full purchase history.
3. For popularity, use COUNT() instead of size().
4. "You" refers to the seller, "I" refers to the user/customer.
5. Use case-insensitive and fuzzy search for text properties.
6. Limit results to prevent overwhelming responses.
7. Always use OPTIONAL MATCH for potentially missing relationships.

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
    input_variables=["question","schema"]
)

response_prompt = PromptTemplate(
    template=RESPONSE_TEMPLATE,
    input_variables=["response"]
)
def cyper_search(query):
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
            raise ValueError("No result from Cypher Chain")  # Force fallback to execute
        
        # Generate a response using the result
        response = llm(response_prompt.format(response=result))
        return response
    
    except Exception as e:
        print(f"Primary query error: {str(e)}")
        
        # Proceed to the fallback query
        # try:
        #     fallback_query = f"""
        #     MATCH (p:Product)
        #     WHERE toLower(p.title) CONTAINS toLower('{query}')
        #     OR toLower(p.description) CONTAINS toLower('{query}')
        #     RETURN p.title AS title, p.description AS description, p.price AS price
        #     LIMIT 5
        #     """ 

        #     # there is no such run method etc need to check again
        #     fallback_result = graph.run(fallback_query).data()
            
        #     if fallback_result:
        #         fallback_response = "I found some products that might be relevant:\n"
        #         for item in fallback_result:
        #             fallback_response += f"- {item['title']}: {item['description']} (Price: {item['price']})\n"
        #         return fallback_response
        #     else:
        #         return "I'm sorry, but I couldn't find any information related to your query. Can you please try asking in a different way or provide more details?"
        
    #  except Exception as e:
    #        print(f"Fallback query error: {str(e)}")
    # return "I'm having trouble accessing the product information right now. Please try again later or contact customer support if the issue persists."




# Define a Pydantic model for the request body
class QueryInput(BaseModel):
    query: str

def search_similar_products(graph, query_embedding, top_k=2, category=None):

    # here just add the extra file of seller id 
    # one more problem is our attributs are stringitied json not pure text so detecting one of the idea is
    # take the attributes based on catergory and use tempalting and then make it to meaning full sentence 
    # suppose elctronics then this is ${ram} ${feautres are} etc.....


#  WHERE product.isAvailable = true  
#   WHERE product.isAvailable = true
    #AND ($category IS NULL OR product.category = $category)
# this is condition is removed to for testing after that add some other condition like 
# match with a partivualr seller etc 
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
        result = graph.query(query, params=params)
    except Exception as e:
        print(f"Error querying graph: {str(e)}")
        result = []
    return result


# Define the POST route to process the query
@app.post("/")
async def brain(input_data: QueryInput):
    print('Received request')
    query = input_data.query
    try:
        # Detect query intent
        detected_intent = query_detection_chain.invoke({"query": query})
        # cyper_search(query)


        # Generate query embedding
        print("embedding started")
        query_embedding = model.encode(query)
        print("embedding finished")

        print(query_embedding)
         
        print(graph.query("""
        CALL db.index.vector.queryNodes("product_embedding_index", 1, $query_embedding) 
        YIELD node, score 
        RETURN node, score
        """, 
        {"query_embedding": query_embedding}))

        
        # Search for similar products the resulti is a an obj/json

        results = search_similar_products(graph, query_embedding, top_k=1, category="")
        
        return {
            "category": detected_intent.strip(),
            "similar_products": results
        }

        # till here similairty search




    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
