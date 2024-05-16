import pinecone
import openai

# Initialize Pinecone
pinecone.init(api_key='YOUR_PINECONE_API_KEY', environment='us-west1-gcp')  # Specify your Pinecone environment if needed

# Create or connect to an existing Pinecone index
index_name = 'gpt4-index'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768)  # Assuming embedding dimension is 768
index = pinecone.Index(index_name)

# Initialize OpenAI API
openai.api_key = 'YOUR_OPENAI_API_KEY'

def embed_text(text):
    # Use OpenAI's API to get the text embedding (assuming GPT-4 or another model that supports embeddings)
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return response['data'][0]['embedding']

# Example text to embed
texts = ["Hello world!", "How are you?"]

# Embed the texts
embeddings = [embed_text(text) for text in texts]

# Create vectors with unique IDs
vectors = [{"id": f"text_{i}", "values": embedding} for i, embedding in enumerate(embeddings)]

# Upsert vectors into the Pinecone index
index.upsert(vectors)

# Querying the Pinecone index with a new text
query_text = "Greetings!"
query_embedding = embed_text(query_text)

# Perform the query
results = index.query(queries=[query_embedding], top_k=5)
print(results)
