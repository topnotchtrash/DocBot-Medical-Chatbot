import os
import streamlit as st
from LLM_Connect_Memory import create_qa_chain

# Set up Streamlit page config
st.set_page_config(page_title="DocBot - Medical Chatbot", page_icon="🩺", layout="centered")

# Cache chatbot initialization
@st.cache_resource
def load_qa_chain():
    return create_qa_chain()

qa_chain = load_qa_chain()

# Apply Custom CSS for chat-style UI
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .stTextInput > div > div > input {
            border-radius: 15px;
            padding: 12px;
            background-color: #f9f9f9;
            border: 2px solid #4CAF50;
            font-size: 16px;
            width: 100%;
        }
        .stButton>button {
            border-radius: 10px;
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .chat-box {
            background-color: #f9f9f9;
            padding: 12px;
            border-radius: 12px;
            font-size: 16px;
            line-height: 1.5;
            color: black;
            border-left: 4px solid #4CAF50;
            margin-bottom: 10px;
        }
        .user-message {
            text-align: right;
            color: #333;
            background-color: #d4fcd4;
            padding: 10px;
            border-radius: 12px;
            margin-bottom: 10px;
        }
        .bot-message {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 12px;
            border-left: 4px solid #4CAF50;
            margin-bottom: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .footer {
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# Styled title
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🩺 DocBot - Your Medical Assistant</h1>
    <p style='text-align: center; font-size: 18px;'>Ask me a medical question, and I'll provide a clear and helpful response.</p>
""", unsafe_allow_html=True)

# Guide for prompts
st.markdown("""
### 💡 **Try Asking DocBot:**
- What are the early signs of **diabetes**?
- How can I boost my **immune system naturally**?
- What’s the best way to **lower cholesterol**?
- How can I improve my **sleep quality**?
- What are common **vitamin deficiencies**?""")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role, content = message["role"], message["content"]
    if role == "user":
        st.markdown(f"""<div class='user-message'><b>👤 You:</b> {content}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='bot-message'><b>🤖 DocBot:</b> {content}</div>""", unsafe_allow_html=True)

# User input
user_query = st.text_input("Ask me a medical question:", placeholder="Type your medical query here...")

if st.button("Get Answer") and user_query:
    with st.spinner("Fetching answer..."):
        response = qa_chain.invoke({"query": user_query})

    formatted_response = response["result"].replace("\n", "<br>")

    # Store user query and response in session state
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "bot", "content": formatted_response})

    # Display the chatbot response
    st.markdown(f"""<div class='bot-message'><b>🤖 DocBot:</b> {formatted_response}</div>""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <p class='footer'>Made with ❤️ for better healthcare assistance - Created by <a href='https://shubhangportfolio.vercel.app/' target='_blank'>Shubhang</a></p>
""", unsafe_allow_html=True)
