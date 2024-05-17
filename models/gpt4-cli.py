import csv
from openai import OpenAI
import os

def load_csv(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data


openai_client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("open_ai_key"),
)


# Define your conversation

def submit_query_to_openai(sentences, query):
    # Combine the relevant columns from all rows

    # Form the comprehensive message for OpenAI
    message = f"{query}\nData:\n{sentences}"
    
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": message},
]
    response = openai_client.chat.completions.create(
        model="gpt-4",  # Specify the GPT-4 model
        messages=messages,
        max_tokens=1000
    )

    return response

def describe_data(data, column_descriptions):
    sentences = []
    for row in data:
        sentence = "In this data, "
        for col_name, col_desc in column_descriptions.items():
            sentence += f"{row[col_name]} {col_desc}, "
        sentence = sentence[:-2] + "."
        sentences.append(sentence)
    return sentences

def main():
    file_path = input("Enter the CSV file path: ")
    query = input("Enter your query: ")
    column_descriptions = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for col_name in reader.fieldnames:
            col_desc = input(f"Enter a brief description of the '{col_name}' column: ")
            column_descriptions[col_name] = col_desc

    data = load_csv(file_path)
    sentences = describe_data(data, column_descriptions)
    response = submit_query_to_openai(sentences, query)

    print(f"Response:\n{response}")
    
if __name__ == "__main__":
    main()
