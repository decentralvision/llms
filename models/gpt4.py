# import faiss
import numpy as np
from openai import OpenAI
import fitz  # PyMuPDF
import pandas as pd
import torch, sys, os
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

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
pdf_text = "Alex sold 5 items in January. Dan sold 8 items in January. Dan also sold 9 items in February. Jeremy sold 6 items in March." # extract_text_from_pdf(pdf_path)

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

model_name='gpt-4'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_answer(query, context, max_length=50):
    # Combine query with context
    input_text = context + "\n\n" + query

    # Tokenize the input query
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    # Generate the answer
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


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
# prompt = "Once upon a time"
# generated_text = query_gpt4(prompt)
# print(generated_text)



def query_model(query, sentences, sentence_embeddings, top_k=5):
    # Embed the query
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # Compute cosine similarity between the query and the sentences
    cos_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

    # Ensure top_k is not greater than the number of sentences
    top_k = min(top_k, len(sentences))

    # Find the top-k most similar sentences
    top_results = torch.topk(cos_scores, k=top_k)

    # Retrieve and return the top-k sentences
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append((sentences[idx], score.item()))

    return results


# Example usage
query_to_answer = "Who sold the most items in January?"
answer = generate_answer(query_to_answer, combined_text)
print(answer)

exit(0)

# Query the model
query = "Who sold the most items?"
results = query_model(query, sentences, sentence_embeddings)

# Print all items in the results array

print(results)

for result in results:
    print(f"Score: {result[1]:.4f} - Sentence: {result[0]}")
