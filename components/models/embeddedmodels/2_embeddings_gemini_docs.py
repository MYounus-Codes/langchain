from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",)

documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "It enables developers to build applications that can understand and generate natural language.",
    "LangChain provides tools for prompt management, memory, and integration with external data sources.",
]

result = embedding.embed_documents(documents)
print(str(result))
