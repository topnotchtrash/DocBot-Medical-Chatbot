# Define paths
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Step 1: Load Raw PDF(s)
def load_pdf_files(data_path):
    """
    Load all PDF files from a directory and extract text.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: Directory '{data_path}' does not exist.")
    
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("Warning: No PDFs found in the directory.")
    
    return documents

# Step 2: Chunk the Text
def chunk_documents(documents, chunk_size=512, chunk_overlap=50):
    """
    Splits documents into smaller chunks for efficient embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# Step 3: Create Vector Embeddings
def get_embeddings():
    """
    Loads a Hugging Face model for embedding creation.
    """
    print("✅ Step 3: Generating embeddings...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store Embeddings in FAISS
def store_embeddings_faiss(chunks, embed_model):
    """
    Creates FAISS index for storing and retrieving embeddings.
    """
    if not chunks:
        print("⚠️ Warning: No chunks found. Skipping FAISS storage.")
        return None
    
    print("✅ Step 4: Creating FAISS index...")
    db = FAISS.from_documents(chunks, embed_model)
    
    print(f"✅ Saving FAISS index to {DB_FAISS_PATH}...")
    db.save_local(DB_FAISS_PATH)
    
    print("✅ FAISS index saved successfully!")
    return db

# Example Usage
if __name__ == "__main__":
    print("Loading PDFs...")
    docs = load_pdf_files(DATA_PATH)
    print(f"Loaded {len(docs)} documents from PDFs.")
    
    print("Chunking documents...")
    chunked_docs = chunk_documents(docs)
    print(f"Created {len(chunked_docs)} text chunks.")
    
    print("Generating embeddings...")
    embed_model = get_embeddings()
    faiss_index = store_embeddings_faiss(chunked_docs, embed_model)
    
    if faiss_index:
        print("🚀 FAISS embedding storage process completed!")
    else:
        print("⚠️ FAISS storage skipped due to missing chunks.")