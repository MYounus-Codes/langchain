from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
response = model.invoke("What is the capital of Pakistan?")
# print(response) # Printing the full response
print(response.content) # Printing only the content of the response
