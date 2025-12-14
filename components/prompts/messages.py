from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key="AIzaSyDBoxmlD1e8NJ8aUPzKRd6INRQGzLRkdzs"
)

messages = [
    SystemMessage(content="You are a helpful assistant.", additional_kwargs={"role": "assistant"}),
    HumanMessage(content="Hello, who won the world series in 2020?"),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)

