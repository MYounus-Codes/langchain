from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",)

result = embedding.embed_query("My name is Muhammad Younus")
print(str(result))
