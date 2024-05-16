import os
import google.generativeai as gemini
import fitz  # PyMuPDF
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Set up your Google Gemini API key
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"

# Initialize the Gemini Client
gemini.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from CSV
def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    summary_column = "Description"
    creator_column = "Creator"
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

pdf_path = "/Users/alex.mills/Downloads/Invitation_letter-Google-Docs.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

csv_path = "/Users/alex.mills/Downloads/Jira.csv"
csv_text = extract_text_from_csv(csv_path)

# Combine extracted text
combined_text = pdf_text + "\n" + csv_text

# Load a pre-trained model for embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2')
sentences = combined_text.split('\n')
sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)

# Define a function to query Google Gemini
def query_gemini(prompt):
    response = gemini.generate_text(
        model="gemini-model",
        prompt=prompt,
        # max_tokens=150
    )
    return response['text'].strip()

# Generate some text
prompt = "Once upon a time"
generated_text = query_gemini(prompt)
print(generated_text)

def query_model(query, sentences, sentence_embeddings, top_k=5):
    # Embed the query
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # Compute cosine similarity between the query and the sentences
    cos_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

    # Find the top-k most similar sentences
    top_results = torch.topk(cos_scores, k=top_k)

    # Retrieve and return the top-k sentences
    results = [(sentences[idx], score.item()) for score, idx in zip(top_results[0], top_results[1])]
    return results

# Query the model
query = "how many jira tickets did alex mills create?"
results = query_model(query, sentences, sentence_embeddings)

# Print all items in the results array
for result in results:
    print(f"Score: {result[1]:.4f} - Sentence: {result[0]}")
