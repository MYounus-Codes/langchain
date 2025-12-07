from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file
model = ChatAnthropic(model="claude-2", temperature=0.7) # you can change the model
response = model.invoke("What is the capital of Pakistan?")
print(response.content)  # Printing only the content of the response
