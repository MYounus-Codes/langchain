from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file

# model = ChatGoogleGenerativeAI(model='gemini-2.5-flash') ## Default temperature is 0.0
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.3)  
# Temperature is use to control the randomness of the output
# You can also provide the max_completions_tokens parameter to limit the response length
response = model.invoke("What is the capital of Pakistan?")
# print(response) # Printing the full response
print(response.content) # Printing only the content of the response
