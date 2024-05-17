import os
import fitz  # PyMuPDF
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import google.generativeai as gemini
import google.generativeai as genai

# Set up your Google Gemini API key
gemini.configure(api_key=os.getenv("gemini_api_key"))

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
pdf_text = "Alex sold 5 items in January. Dan sold 8 items in January. Dan also sold 9 items in February. Jeremy sold 6 items in March." # extract_text_from_pdf(pdf_path)

csv_path = "/Users/alex.mills/Downloads/Jira.csv"
csv_text = "" # extract_text_from_csv(csv_path)

# Combine extracted text
combined_text = pdf_text + "\n" + csv_text

# Load a pre-trained model for embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2')
sentences = combined_text.split('\n')
sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)

# Define a function to parse sales data from text
def parse_sales_data(sentences):
    sales_data = {}
    for sentence in sentences:
        parts = sentence.split()
        if "sold" in parts:
            try:
                name = parts[0]
                sold_index = parts.index("sold")
                items_sold = int(parts[sold_index + 1])
                if name in sales_data:
                    sales_data[name] += items_sold
                else:
                    sales_data[name] = items_sold
            except (IndexError, ValueError):
                continue
    return sales_data

# Define a function to determine who sold the most
def who_sold_the_most(sales_data):
    if not sales_data:
        return "No sales data available."
    max_sales = max(sales_data.values())
    top_sellers = [name for name, sales in sales_data.items() if sales == max_sales]
    return top_sellers, max_sales

# Parse sales data
sales_data = parse_sales_data(sentences)

# Determine who sold the most
top_sellers, max_sales = who_sold_the_most(sales_data)
print(f"Top seller(s): {', '.join(top_sellers)} with {max_sales} items sold.")

# Optional: Use Google Gemini to generate a response
model = genai.GenerativeModel('gemini-pro')

def query_gemini(prompt):
    response = model.generate_content("Who sold the most this year?")
    return response

# Example prompt for Google Gemini
prompt = "Based on the sales data, who sold the most items this year?"
gemini_response = query_gemini(prompt)
print(gemini_response)
