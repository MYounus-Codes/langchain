## In today's days, LLMs are not using so much. But still, for demo purpose, we can use them.

from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file

llm = OpenAI(model="gpt-4", temperature=0.7)

result = llm.invoke("What is the capital of Pakistan?")
print(result)

### Output:  The capital of Pakistan is Islamabad. ###

## With minor changes, we can use other LLMs as well.

from langchain_google_genai import GoogleGenerativeAI

llm_gemini = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
result_gemini = llm_gemini.invoke("What is the capital of Pakistan?")

print(result_gemini.content)