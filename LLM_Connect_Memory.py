import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Groq API
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Error: Groq API Key not found. Set it in the .env file.")

# Initialize Groq LLM Model
def load_llm():
    print("Loading Groq AI Model...")
    return ChatGroq(model_name="llama3-8b-8192", api_key=GROQ_API_KEY)

# Load FAISS database for retrieval
def load_faiss_index():
    print("Loading FAISS index...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)

# Define a prompt template
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

# Create the retrieval-based QA chatbot
def create_qa_chain():
    llm = load_llm()
    db = load_faiss_index()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    prompt = set_custom_prompt()
    print("Creating QA Chain...")
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,  # Do not return sources
        chain_type_kwargs={"prompt": prompt}
    )
