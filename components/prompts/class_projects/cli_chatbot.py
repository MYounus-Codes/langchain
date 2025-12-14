from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    # api_key=os.getenv("GOOGLE_GENAI_API_KEY"),
    api_key="AIzaSyDBoxmlD1e8NJ8aUPzKRd6INRQGzLRkdzs",
)

chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    user = input("You: ")
    chat_history.append(HumanMessage(content=user))
    if user.lower() in ["exit", "quit"]:
        print("Exiting chat...")
        break 
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI:", response.content)

print("Chat ended.")
print("Final chat history:", chat_history)
