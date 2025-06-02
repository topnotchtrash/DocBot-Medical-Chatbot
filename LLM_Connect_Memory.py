import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Groq API
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from data_fetcher import DataFetcher
from typing import Dict, Any, Optional, List
from langchain.schema import Document

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

def extract_medical_topic(llm, question: str) -> Optional[str]:
    """
    Use LLM to extract the main medical topic/condition from the question.
    """
    topic_prompt = PromptTemplate(
        template="""
        Extract the main medical condition or health topic from the question. 
        Return ONLY the topic name, nothing else. If no clear medical topic is found, return "None".
        
        Question: {question}
        
        Topic:""",
        input_variables=["question"]
    )
    
    response = llm.invoke(topic_prompt.format(question=question))
    topic = response.content.strip()
    return None if topic.lower() == "none" else topic

def update_faiss_with_new_data(topic: str, faiss_db: FAISS) -> bool:
    """
    Fetch new data for a topic and add it to the FAISS index.
    Returns True if successful, False otherwise.
    """
    try:
        print(f"🔄 Fetching new data for topic: {topic}")
        fetcher = DataFetcher([topic])
        topic_data = fetcher.fetch_topic_data()
        
        if not topic_data or topic not in topic_data:
            return False
            
        # Convert the new data to documents
        new_documents = []
        content = topic_data[topic]
        
        # Process health articles
        for article in content.get('health_articles', []):
            full_content = f"Title: {article['title']}\n\n{article['full_text']}"
            metadata = {
                'topic': topic,
                'type': 'health',
                'url': article['url'],
                'source': 'MedlinePlus',
                'source_type': 'medlineplus'
            }
            new_documents.append(Document(page_content=full_content, metadata=metadata))
        
        # Process drug articles
        for article in content.get('drug_articles', []):
            full_content = f"Title: {article['title']}\n\n{article['full_text']}"
            metadata = {
                'topic': topic,
                'type': 'drug',
                'url': article['url'],
                'source': 'MedlinePlus',
                'source_type': 'medlineplus'
            }
            new_documents.append(Document(page_content=full_content, metadata=metadata))
        
        if new_documents:
            print(f"📥 Adding {len(new_documents)} new documents to FAISS index")
            faiss_db.add_documents(new_documents)
            # Save the updated index
            faiss_db.save_local("vectorstore/db_faiss")
            return True
            
        return False
        
    except Exception as e:
        print(f"❌ Error updating FAISS index: {str(e)}")
        return False

# Define a prompt template
def set_custom_prompt():
    return PromptTemplate(
        template="""
        You are a medical assistant. Use the provided context to answer the question factually.
        If you cannot find enough information in the context to provide a complete answer, respond with exactly "NEED_MORE_CONTEXT" and nothing else.
        Otherwise, provide a clear and concise answer based on the context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "question"]
    )

class MedicalQA:
    def __init__(self):
        self.llm = load_llm()
        self.db = load_faiss_index()
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        retriever = self.db.as_retriever(search_kwargs={"k": 3})
        prompt = set_custom_prompt()
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )

    def answer_question(self, question: str) -> str:
        # First attempt with existing knowledge
        response = self.qa_chain.invoke({"query": question})
        answer = response.get('result', '').strip()

        # If we need more context, try to fetch new data
        if answer == "NEED_MORE_CONTEXT":
            print("🔍 Initial answer insufficient, attempting to fetch more data...")
            topic = extract_medical_topic(self.llm, question)
            
            if topic:
                print(f"📚 Identified topic: {topic}")
                if update_faiss_with_new_data(topic, self.db):
                    print("🔄 Retrying question with updated knowledge...")
                    # Recreate the chain with updated index
                    self.qa_chain = self._create_qa_chain()
                    response = self.qa_chain.invoke({"query": question})
                    answer = response.get('result', '').strip()
                    
                    if answer == "NEED_MORE_CONTEXT":
                        return "I apologize, but I don't have enough information to provide a complete answer to your question, even after searching for more data."
                else:
                    return "I apologize, but I couldn't find additional information about this topic."
            else:
                return "I apologize, but I couldn't identify the medical topic in your question to search for more information."

        return answer

# Create the retrieval-based QA chatbot (for backward compatibility)
def create_qa_chain():
    medical_qa = MedicalQA()
    return medical_qa.qa_chain

# Example usage
if __name__ == "__main__":
    medical_qa = MedicalQA()
    while True:
        question = input("\nEnter your medical question (or 'quit' to exit): ")
        if question.lower() in ['quit', 'exit']:
            break
        print("\n🤖 Answer:", medical_qa.answer_question(question))
