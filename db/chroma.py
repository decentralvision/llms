import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB
vector_db = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Create or get the collection
collection_name = "my_collection"
collection = vector_db.get_or_create_collection(collection_name)

# Initialize the SentenceTransformer model
sentence_transformers_model = SentenceTransformer('all-MiniLM-L6-v2')

def add_data_to_collection(texts, collection):
    # Encode the texts
    vectors = sentence_transformers_model.encode(texts)

    # Prepare data for insertion
    ids = [str(i) for i in range(len(texts))]
    metadata = [{"text": text} for text in texts]

    # Insert data into ChromaDB
    collection.add(
        embeddings=vectors,
        metadatas=metadata,
        ids=ids
    )

# Sample data
texts = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome.",
    # Add more texts as needed
]

# Add data to the collection
add_data_to_collection(texts, collection)

def query_collection(query, collection):
    # Encode the query
    query_vector = sentence_transformers_model.encode(query).tolist()

    # Query the collection
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3  # Number of similar vectors to fetch
    )
    return results

# Verify insertion by querying the collection
query_result = query_collection("What is the capital of France?", collection)
print(query_result)
