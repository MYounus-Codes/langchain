## 1_document_similarity.py
# This script computes the similarity between two text documents using cosine similarity.

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Initialize the Google Generative AI Embeddings model
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

## Prepare the documents
documents = [
    "Cristiano Ronaldo is a professional footballer who plays as a forward for Al Nassr and the Portugal national team.",
    "Lionel Messi is an Argentine professional footballer who plays as a forward for Inter Miami and the Argentina national team.",
    "Virat Kohli is an Indian cricketer and former captain of the India national team.",
    "Babar Azam is a Pakistani cricketer and the captain of the Pakistan national cricket team."
]

# Query
query = "Who is the captain of the Pakistan national cricket team?"

# Generate embeddings for the documents
document_embeddings = embeddings_model.embed_documents(documents)

# Generate embedding for the query
query_embedding = embeddings_model.embed_query(query)

# Compute cosine similarity between the query and each document
similarities = cosine_similarity([query_embedding], document_embeddings)
index, score = sorted(list(enumerate(similarities)), key=lambda x: x[1])[-1]
# Output the most similar document and its similarity score
print("Query:", query)
print(f"Most similar document: {documents[index]}")
print(f"Similarity score: {score}")
