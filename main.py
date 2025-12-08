### main.py

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=api_key)

    user_input = "Explain the theory of relativity in simple terms."
    response = chat.invoke(user_input)

    print("Response from Google Generative AI:")
    print(response.content)
if __name__ == "__main__":
    main()

