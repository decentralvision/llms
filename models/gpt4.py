# import faiss
import numpy as np
from openai import OpenAI
import fitz  # PyMuPDF
import pandas as pd
import torch, sys, os
from sentence_transformers import SentenceTransformer, util

# Set up your OpenAI API key
# openai.api_key = os.getenv('open_ai_key')

openai_client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("open_ai_key"),
)


# Extract text from CSV
def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    print("Columns:", df.columns)  # Print the columns for inspection

    # Identify relevant columns (customize based on your CSV structure)
    summary_column = "Description"
    creator_column = "Creator"

    # Combine relevant text
    combined_text = ""
    for index, row in df.iterrows():
        combined_text += f"{row[creator_column]} created a ticket: {row[summary_column]}\n"

    return combined_text

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

pdf_path = "/Users/alex.mills/Downloads/sales.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

csv_path = "/Users/alex.mills/Downloads/Jira.csv"
csv_text = "" # extract_text_from_csv(csv_path)

# Combine extracted text
combined_text = pdf_text + "\n" + csv_text

# Load a pre-trained model for embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Split the combined text into sentences or chunks
sentences = combined_text.split('\n')

# Create embeddings for the sentences
sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)

# Define a function to query ChatGPT-4
def query_gpt4_old(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def query_gpt4(prompt):
    chat_completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    )
    return chat_completion

# Generate some text
prompt = "Once upon a time"
generated_text = query_gpt4(prompt)
print(generated_text)

def query_model(query, sentences, sentence_embeddings, top_k=5):
    # Embed the query
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # Compute cosine similarity between the query and the sentences
    cos_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

    # Find the top-k most similar sentences
    top_results = torch.topk(cos_scores, k=top_k)

    # Retrieve and return the top-k sentences
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append((sentences[idx], score.item()))

    return results

# Query the model
query = "how many jira tickets did alex mills create?"
results = query_model(query, sentences, sentence_embeddings)

# Print all items in the results array
for result in results:
    print(f"Score: {result[1]:.4f} - Sentence: {result[0]}")
