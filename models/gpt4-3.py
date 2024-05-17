import pandas as pd
import openai

# Read the CSV file
df = pd.read_csv('sales_data.csv')

# Format the data
data = df.to_dict(orient='records')
formatted_data = []
for row in data:
    formatted_data.append(f"{row['Name']} sold {row['Items Sold']} items in {row['Month']}.")

# Create the conversation context
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Here is the sales data:"},
]

# Add the formatted data to the messages
for data_entry in formatted_data:
    messages.append({"role": "user", "content": data_entry})

# Ask the question based on the data
messages.append({"role": "user", "content": "Who sold the most items overall?"})

# Generate a response
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    max_tokens=1000
)

print(response['choices'][0]['message']['content'])
