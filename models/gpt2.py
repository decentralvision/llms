from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import sys
import fitz  # PyMuPDF
import pandas as pd
from sentence_transformers import SentenceTransformer, util


def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    print("Columns:", df.columns)  # Print the columns for inspection

    # Identify relevant columns (customize these column names based on your CSV structure)
    summary_column = "Description"
    creator_column = "Creator"

    # # Try to infer the columns automatically
    # for col in df.columns:
    #     if 'summary' in col.lower():
    #         summary_column = col
    #     if 'creator' in col.lower() or 'author' in col.lower():
    #         creator_column = col
    #
    # if not summary_column or not creator_column:
    #     raise ValueError("Could not infer necessary columns from the CSV file")

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

pdf_path = "/Users/alex.mills/Downloads/Invitation_letter-Google-Docs.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

# Extract text from CSV
def extract_text_from_csv_1(csv_path):
    df = pd.read_csv(csv_path)
    # Convert the DataFrame to text (customize based on your CSV structure)
    text = df.to_string(index=False)
    return text

csv_path = "/Users/alex.mills/Downloads/Jira.csv"
csv_text = extract_text_from_csv(csv_path)

# Combine extracted text
combined_text = pdf_text + "\n" + csv_text

# Load a pre-trained model for embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Split the combined text into sentences or chunks
sentences = combined_text.split('\n')


# Create embeddings for the sentences
sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define a function to generate text
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate some text
prompt = "Once upon a time"
generated_text = generate_text(prompt)
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

for result in results:
    print(f"Score: {result[1]:.4f} - Sentence: {result[0]}")

for result in results:
    print(f"Score: {result[1]:.4f} - Sentence: {result[0]}")