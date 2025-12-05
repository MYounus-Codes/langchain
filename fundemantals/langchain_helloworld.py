from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.5-flash")

print(llm.invoke("Hi there! What is LangChain?"))