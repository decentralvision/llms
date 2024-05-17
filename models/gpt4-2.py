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


# Define your conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Alex sold 5 items in January. Dan sold 8 items in January. Dan also sold 9 items in February. Jeremy sold 6 items in March."},
    {"role": "user", "content": "Who sold the most?"}
]

# Generate a response
response = openai_client.chat.completions.create(
    model="gpt-4",  # Specify the GPT-4 model
    messages=messages,
    max_tokens=1000
)

# Print the response
# answer = response['choices'][0]['message']['content'].strip()
print(response)