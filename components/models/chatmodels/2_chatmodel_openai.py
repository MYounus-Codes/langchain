from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file

model = ChatOpenAI(model="gpt-4", temperature=0.7)
response = model.invoke("What is the capital of Pakistan?")
print(response.content)  # Printing only the content of the response