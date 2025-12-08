from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    # api_key=os.getenv("GOOGLE_GENAI_API_KEY"),
    api_key="AIzaSyDBoxmlD1e8NJ8aUPzKRd6INRQGzLRkdzs",
)

chat_history = []

while True:
    user = input("You: ")
    chat_history.append({"role": "user", "content": user})
    if user.lower() in ["exit", "quit"]:
        print("Exiting chat...")
        break
    response = model.invoke(chat_history)
    chat_history.append({"role": "assistant", "content": response.content})
    print("AI:", response.content)

print("Chat ended.")
print("Final chat history:", chat_history)
