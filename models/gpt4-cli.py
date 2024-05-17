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

def submit_query_to_openai(data, query, columns):
    # Combine the relevant columns from all rows
    combined_data = [ {col: row[col] for col in columns} for row in data ]

    # Form the comprehensive message for OpenAI
    message = f"{query}\nData:\n{combined_data}"
    
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

def main():
    file_path = input("Enter the CSV file path: ")
    query = input("Enter your query: ")
    columns = input("Enter the column names (comma-separated): ").split(',')

    data = load_csv(file_path)
    response = submit_query_to_openai(data, query, columns)

    print(f"Response:\n{response}")

if __name__ == "__main__":
    main()
