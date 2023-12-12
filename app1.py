import requests
from transformers import BertTokenizer, BertModel
import torch
import dotenv, os

dotenv.load_dotenv()

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


# Function to convert text to vector using BERT
def text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().tolist()[0]


# Read text from file
file_path = "./files/resume.txt"  # Replace with your file path
with open(file_path, "r") as file:
    text = file.read()

# Convert text to vector
vector = text_to_vector(text)

# Structure the data for Weaviate
data = {
    "class": "TextDocument",  # Replace with your class name in Weaviate
    "properties": {"content": text, "contentVector": vector},
}

# Replace with your Weaviate instance URL and port
# weaviate_url = 'http://localhost:8080/v1/objects'
import weaviate

auth_config = weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"])

client = weaviate.Client(
    url="https://my-personal-bot-bxmogd3i.weaviate.network",
    auth_client_secret=auth_config,
    #   embedded_options = EmbeddedOptions()
)

# Import data into Weaviate
response = requests.post(client, json=data)
if response.status_code == 200:
    print("Data successfully imported to Weaviate.")
else:
    print(f"Failed to import data: {response.text}")
