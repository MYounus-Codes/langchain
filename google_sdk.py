from google import genai
import os
api_key = os.getenv("GOOGLE_API_KEY")
model = "gemini-2.5-flash"
client = genai.Client(api_key=api_key)

response = client.models.generate_content(model=model, contents="Hello")
print(response.text)