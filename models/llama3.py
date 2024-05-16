from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import fitz  # PyMuPDF
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Specify the model you want to use
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # Adjust to the correct model ID

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# Create a text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Function to generate text using the model
def generate_text(prompt):
    outputs = text_generator(prompt, max_length=150, do_sample=True, temperature=0.7, top_p=0.9)
    return outputs[0]["generated_text"]

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)

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

# Function to query model with embeddings
def query_model(query, sentences, sentence_embeddings, top_k=5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    results = [(sentences[idx], score.item()) for score, idx in zip(top_results[0], top_results[1])]
    return results

# Query the model
query = "how many jira tickets did alex mills create?"
results = query_model(query, sentences, sentence_embeddings)

# Print all items in the results array
for result in results:
    print(f"Score: {result[1]:.4f} - Sentence: {result[0]}")
