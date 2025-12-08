## Install the required library
# - pip install langchain-huggingface
# - pip install sentence-transformers

# 3_embeddings_hf_local.py

# Import the necessary modules
from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Use a local HuggingFace model

## Simple Query Embedding
text = "LangChain is a framework for developing applications powered by language models."
result = embedding.embed_query(text) # Embed the query text
print(str(result))

# ## Document Embedding
# documents = [
#     "LangChain is a framework for developing applications powered by language models.",
#     "It enables developers to build applications that can understand and generate natural language.",
#     "LangChain provides tools for prompt management, memory, and integration with external data sources.",
# ]
# result = embedding.embed_documents(documents)
# print(str(result))
