from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_GENAI_API_KEY"),
)

while True:
    user = input("You: ")
    if user.lower() in ["exit", "quit"]:
        print("Exiting chat...")
        break
    response = model.invoke(user)
    print("AI:", response.content)
    