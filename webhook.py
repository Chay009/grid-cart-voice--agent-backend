from fastapi import FastAPI, HTTPException, Request
from langchain_community.vectorstores import Neo4jVector
import os
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field

app = FastAPI()

# Load environment variables
load_dotenv()

# Initialize Neo4j credentials
neo4j_uri = os.getenv('NEO4J_URI')
neo4j_password = os.getenv('NEO4J_PASSWORD')
neo4j_username = os.getenv('NEO4J_USERNAME')



## move the brain route to here

class HuggingFaceInferenceEmbeddings(BaseModel, Embeddings):
    """HuggingFace Inference API embedding models.

    To use, you should have an API key for the Hugging Face Inference API stored in the HUGGINGFACE_API_KEY environment variable.

    Example:
        .. code-block:: python

            from huggingface_inference_embeddings import HuggingFaceInferenceEmbeddings

            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            hf = HuggingFaceInferenceEmbeddings(model_name=model_name)
    """

    api_key: str = Field(default_factory=lambda: os.getenv('HUGGINGFACE_API_KEY'))
    """Hugging Face API key."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    """Model name to use."""

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def _get_embedding(self, text: str) -> List[float]:
        """Get embeddings for a single piece of text."""
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(api_url, headers=headers, json={"inputs": text})
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            error_msg = f"Error calling Hugging Face API: {str(e)}"
            if response.text:
                error_msg += f"\nResponse body: {response.text}"
            raise ValueError(error_msg)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using the Hugging Face Inference API.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using the Hugging Face Inference API.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._get_embedding(text)

# Usage example:
hf_inference_embeddings = HuggingFaceInferenceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
def initialize_vector_index():
    """Initialize the vector index in Neo4j using embeddings from Hugging Face."""
    try:
        print("Initializing vector index...")
        neo4j_vector = Neo4jVector.from_existing_graph(
            embedding=hf_inference_embeddings,  # Use the custom Hugging Face embedding
            node_label="Product",
            embedding_node_property="embedding",
            text_node_properties=["title", "description", "attributes", "category", "brand"],
            index_name="product_embedding_index",
            search_type="hybrid",
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        print("Vector index initialized.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing vector index: {str(e)}")

@app.post("/webhook/initialize-vector-index")
async def initialize_vector_index_route(request: Request):
    """Endpoint to initialize vector index."""
    initialize_vector_index()
    print("Created embeddings for non-initialized vectors")
    return {"message": "Vector index initialized successfully."}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
