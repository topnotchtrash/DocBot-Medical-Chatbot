import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Error: Groq API Key not found. Set it in the .env file.")

HUGGINGFACE_REPO_ID = "HuggingFaceH4/zephyr-7b-alpha"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load the LLM model from Hugging Face
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        token=GROQ_API_KEY,
        temperature=0.7,
        top_p=0.9,
        max_length=512
    )

# Load FAISS index for retrieval
def load_faiss_index():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Define a prompt template for responses
def set_custom_prompt():
    return PromptTemplate(
        template="""
        You are a medical assistant. Use the provided context to answer the question factually.
        - If the context does not contain the answer, say you don't know.
        - Keep responses clear, concise, and medically accurate.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """,
        input_variables=["context", "question"]
    )

# Create a retrieval-based QA chatbot
def create_qa_chain():
    llm = load_llm()
    db = load_faiss_index()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    prompt = set_custom_prompt()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

# Set up Streamlit page config
st.set_page_config(page_title="DocBot - Medical Chatbot", page_icon="🩺", layout="centered")

# Cache chatbot initialization
@st.cache_resource
def load_qa_chain():
    return create_qa_chain()

qa_chain = load_qa_chain()

# Apply Custom CSS for Dark Mode Fix & UI Enhancements
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .main {
            background-color: #ffffff;
            color: black;
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
            color: #ffffff;
            background-color: #4CAF50;
            padding: 10px;
            border-radius: 12px;
            margin-bottom: 10px;
        }
        .bot-message {
            background-color: #333333; /* Dark mode fix */
            padding: 10px;
            border-radius: 12px;
            border-left: 4px solid #4CAF50;
            margin-bottom: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            color: #ffffff; /* Light text for better visibility */
        }
        .footer {
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: #bbb;
        }
    </style>
""", unsafe_allow_html=True)

# Styled title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🩺 DocBot - Your Medical Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Ask me a medical question, and I'll provide a clear and helpful response.</p>", unsafe_allow_html=True)

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
st.markdown("<p class='footer'>Made with ❤️ for better healthcare assistance</p>", unsafe_allow_html=True)
