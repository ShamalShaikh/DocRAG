import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name="pdf_rag",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity for better matching
)
