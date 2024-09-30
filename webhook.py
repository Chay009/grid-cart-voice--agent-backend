
from fastapi import FastAPI, HTTPException, Request
from langchain_community.vectorstores import Neo4jVector
app = FastAPI()
# FastAPI route to initialize the vector index
# Function to initialize or update the vector index

import os
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
# Load environment variables
load_dotenv()
# Initialize Neo4j and Groq API credentials
neo4j_uri = os.getenv('NEO4J_URI')
neo4j_password = os.getenv('NEO4J_PASSWORD')
neo4j_username = os.getenv('NEO4J_USERNAME')
# Initialize HuggingFaceEmbeddings

# Initialize the embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hug_face_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
def initialize_vector_index():

    # this function can't be moved to js it should be in python 
    try:
        print("Initializing vector index...")
        Neo4jVector.from_existing_graph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
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