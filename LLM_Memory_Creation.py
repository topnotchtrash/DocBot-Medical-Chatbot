import os
import json
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Define paths
DATA_PATH = "data/"
JSON_PATH = "topic_article_store.json"
DB_FAISS_PATH = "vectorstore/db_faiss"

def faiss_index_exists() -> bool:
    """
    Check if FAISS index files exist.
    """
    index_files = ["index.faiss", "index.pkl"]
    return all(os.path.exists(os.path.join(DB_FAISS_PATH, f)) for f in index_files)

def load_existing_faiss(embed_model) -> FAISS:
    """
    Load existing FAISS index if it exists, allowing trusted deserialization.
    """
    print("✅ Loading existing FAISS index...")
    return FAISS.load_local(DB_FAISS_PATH, embed_model, allow_dangerous_deserialization=True)

def load_pdf_files(data_path: str) -> List[Document]:
    """
    Load all PDF files from a directory and extract text.
    """
    if not os.path.exists(data_path):
        print(f"⚠️ Warning: Directory '{data_path}' does not exist.")
        return []
    
    print("📄 Loading PDFs...")
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    for doc in documents:
        doc.metadata.update({
            'source_type': 'pdf',
            'source': os.path.basename(doc.metadata.get('source', ''))
        })
    
    print(f"✅ Loaded {len(documents)} documents from PDFs.")
    return documents

def load_json_data(json_path: str) -> List[Document]:
    """
    Load medical articles from JSON and convert to Documents.
    """
    if not os.path.exists(json_path):
        print(f"⚠️ Warning: File '{json_path}' does not exist.")
        return []
    
    print("📄 Loading JSON data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    
    for topic, content in data.items():
        for article in content.get('health_articles', []):
            full_content = f"Title: {article['title']}\n\n{article['full_text']}"
            metadata = {
                'topic': topic,
                'type': 'health',
                'url': article['url'],
                'source': 'MedlinePlus',
                'source_type': 'medlineplus'
            }
            documents.append(Document(page_content=full_content, metadata=metadata))
        
        for article in content.get('drug_articles', []):
            full_content = f"Title: {article['title']}\n\n{article['full_text']}"
            metadata = {
                'topic': topic,
                'type': 'drug',
                'url': article['url'],
                'source': 'MedlinePlus',
                'source_type': 'medlineplus'
            }
            documents.append(Document(page_content=full_content, metadata=metadata))
    
    print(f"✅ Loaded {len(documents)} articles from JSON.")
    return documents

def chunk_documents(documents: List[Document], chunk_size=512, chunk_overlap=50):
    """
    Splits documents into smaller chunks for efficient embedding.
    """
    if not documents:
        return []
        
    print("🧩 Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks.")
    return chunks

def get_embeddings():
    """
    Loads a Hugging Face model for embedding creation.
    """
    print("🧠 Loading embedding model...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def store_embeddings_faiss(chunks: List[Document], embed_model, existing_db=None):
    """
    Creates or updates FAISS index for storing and retrieving embeddings.
    """
    if not chunks:
        print("⚠️ Warning: No chunks found. Skipping FAISS storage.")
        return existing_db if existing_db else None
    
    print("💾 Creating/Updating FAISS index...")
    
    if existing_db:
        existing_db.add_documents(chunks)
        db = existing_db
    else:
        db = FAISS.from_documents(chunks, embed_model)
    
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    
    print("✅ FAISS index saved successfully!")
    return db

# Main execution
if __name__ == "__main__":
    try:
        embed_model = get_embeddings()
        existing_db = None
        pdf_documents = []

        if faiss_index_exists():
            print("📚 Found existing FAISS index")
            existing_db = load_existing_faiss(embed_model)
        else:
            print("🆕 No existing FAISS index found, will create new one")
            pdf_documents = load_pdf_files(DATA_PATH)

        json_documents = load_json_data(JSON_PATH)
        documents_to_process = json_documents + pdf_documents

        if not documents_to_process:
            print("❌ No documents found to process!")
            exit(1)
            
        chunked_docs = chunk_documents(documents_to_process)
        faiss_index = store_embeddings_faiss(chunked_docs, embed_model, existing_db)
        
        if faiss_index:
            print("🚀 FAISS embedding storage process completed successfully!")
            print(f"📊 Stats:")
            print(f"   - PDF documents: {len(pdf_documents)}")
            print(f"   - JSON articles: {len(json_documents)}")
            print(f"   - Total chunks: {len(chunked_docs)}")
        else:
            print("⚠️ FAISS storage failed.")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
