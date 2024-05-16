from openai import OpenAI
import fitz  # PyMuPDF
import pandas as pd
import torch, sys, os
from sentence_transformers import SentenceTransformer, util


from openai import OpenAI
import fitz  # PyMuPDF
import torch, sys, os
from sentence_transformers import SentenceTransformer, util

# Set up your OpenAI API key
# openai.api_key = os.getenv('open_ai_key')

openai_client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("open_ai_key"),
)



# Initialize the SentenceTransformer model
sentence_transformers_model = SentenceTransformer('all-MiniLM-L6-v2')

# Assume you have some vector database instance
vector_db = ...  # replace with actual initialization of your vector database

def answer_question(question, response_from_vector_search):
    # Combine the response from the vector search into a single context string
    context = "\n".join(response_from_vector_search)

    # Construct the prompt for the OpenAI API
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    response = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
    )

    # # Get the response from OpenAI
    # response = openai.Completion.create(
    #     engine="text-davinci-003",
    #     prompt=prompt,
    #     max_tokens=150,
    #     n=1,
    #     stop=None,
    #     temperature=0.7,
    # )
    return response.choices[0].text.strip()

def answer_question_with_data(question):
    vector = encode_question(question)
    response_from_vector_search = [d['text'] for d in find_similar_vectors(vector)]
    return answer_question(question, response_from_vector_search)

def encode_question(q):
    return sentence_transformers_model.encode(q)

def find_similar_vectors(vector):
    # Assuming vector_db.find_similar_vectors(vector) returns a list of dict objects
    # with 'vector' and 'text' keys
    return vector_db.find_similar_vectors(vector)
