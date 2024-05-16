import os

from pinecone import Pinecone
# from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer

pc = Pinecone(api_key=os.getenv('pinecone_api_key'))
index = pc.Index("index-name")

# Initialize Pinecone with the API key
# pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='us-west1-gcp')

# Define the index name
index_name = 'my-index'

# Create the index (only needed once)
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=384)  # dimension should match your embeddings

# Connect to the index
vector_db = pinecone.Index(index_name)

# Initialize the SentenceTransformer model
sentence_transformers_model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample data
texts = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome.",
    # Add more texts as needed
]

# Encode the texts
vectors = sentence_transformers_model.encode(texts)

# Prepare data for insertion
data_to_insert = [
    {"id": str(i), "values": vector.tolist(), "metadata": {"text": text}}
    for i, (vector, text) in enumerate(zip(vectors, texts))
]

# Insert data into Pinecone
vector_db.upsert(vectors=data_to_insert)

# Verify insertion by querying the index
query_vector = sentence_transformers_model.encode("What is the capital of France?").tolist()
response = vector_db.query(
    queries=[query_vector],
    top_k=3,  # Number of similar vectors to fetch
    include_metadata=True
)

print(response)
