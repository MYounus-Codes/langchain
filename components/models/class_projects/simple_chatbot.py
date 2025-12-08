from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
st.header("Simple Chatbot using Langchain and Google Gemini")
user_input = st.text_input("Ask me anything...")
if st.button("Send"):
    # st.balloons()
    response = model.invoke(user_input)
    st.write(response.content)
