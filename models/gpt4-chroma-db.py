from openai import OpenAI
import fitz  # PyMuPDF
import pandas as pd
import torch, os
from sentence_transformers import SentenceTransformer, util
import chromadb
from chromadb.config import Settings

# Set up your OpenAI API key
openai_client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("open_ai_key"),
)


# Initialize the SentenceTransformer model
sentence_transformers_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
vector_db = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Assume you have already added your vectors to the collection
collection = vector_db.get_collection("my_collection")

def find_similar_vectors(vector):
    results = collection.query(
        query_embeddings=[vector],
        n_results=10  # Number of similar vectors to fetch
    )
    return [{"vector": res["embedding"], "text": res["metadata"]["text"]} for res in results["documents"]]

def encode_question(q):
    return sentence_transformers_model.encode(q)

def answer_question(question, response_from_vector_search):
    # Combine the response from the vector search into a single context string
    context = "\n".join(response_from_vector_search)

    # Construct the prompt for the OpenAI API
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    # Get the response from OpenAI
    # response = openai.Completion.create(
    #     engine="text-davinci-003",
    #     prompt=prompt,
    #     max_tokens=150,
    #     n=1,
    #     stop=None,
    #     temperature=0.7,
    # )
    response = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
    )
    return response.choices[0].text.strip()

def answer_question_with_data(question):
    vector = encode_question(question)
    response_from_vector_search = [d['text'] for d in find_similar_vectors(vector)]
    return answer_question(question, response_from_vector_search)
