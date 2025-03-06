# Load environment variables
import os
from dotenv import load_dotenv

# Explicitly specify the .env path if necessary
env_loaded = load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("❌ ERROR: Hugging Face API Token (HF_TOKEN) not found! Set it in Streamlit Secrets or .env")


HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load the LLM model from Hugging Face
def load_llm():
    print("Loading Mistral-7B Model...")
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        token=HF_TOKEN,
        temperature=0.7,
        top_p=0.9,
        max_length=512
    )

# Load FAISS index for retrieval
def load_faiss_index():
    print("Loading FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Define a structured prompt template
def set_custom_prompt():
    return PromptTemplate(
        template=(
            """
            You are a medical assistant. Use the provided context to answer factually.
            - If the context does not contain the answer, say you don't know.
            - Keep responses clear, concise, and medically accurate.
            - Use bullet points where necessary.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            """
        ),
        input_variables=["context", "question"]
    )

# Format chatbot response for readability
def format_chatbot_response(text):
    text = text.replace("\n", "\n\n")  # Add spacing for readability
    text = re.sub(r"(\d+)\.", r"\n**\1.**", text)  # Bold numbers for lists
    return text.replace("-", "•")  # Convert dashes to bullet points

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
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

if __name__ == "__main__":
    qa_chain = create_qa_chain()
    while True:
        user_query = input("\nWrite Query Here: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        response = qa_chain.invoke({"query": user_query})
        print("\nChatbot Response:\n", format_chatbot_response(response["result"]))
        print("\nSources Used:")
        for doc in response["source_documents"]:
            print(f"- {doc.metadata['title']}, Page {doc.metadata['page']}")
