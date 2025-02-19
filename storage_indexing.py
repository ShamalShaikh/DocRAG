"""
Storage & Indexing Module for RAG System.

This module provides functionality for generating embeddings from Markdown content
and storing them in a vector database along with metadata for efficient retrieval.
It supports multiple vector database backends through a modular architecture.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Data class representing a document with its content and metadata."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class VectorDBInterface(ABC):
    """Abstract base class defining the interface for vector database implementations."""
    
    @abstractmethod
    def initialize(self, dimension: int) -> None:
        """Initialize the vector database with the specified dimension."""
        pass
    
    @abstractmethod
    def store(self, documents: List[Document]) -> bool:
        """Store documents and their embeddings in the vector database."""
        pass
    
    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Query the vector database for similar documents."""
        pass
    
    @abstractmethod
    def delete(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector database."""
        pass

class PineconeDB(VectorDBInterface):
    """Pinecone vector database implementation."""
    
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        namespace: str = ""
    ):
        """
        Initialize PineconeDB connection.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            namespace: Optional namespace for the index
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.namespace = namespace
        self.index = None
        
    def initialize(self, dimension: int) -> None:
        """
        Initialize Pinecone index with the specified dimension.
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=self.api_key)
        
        # Create index if it doesn't exist
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec()  # Using serverless for better scalability
            )
            
        self.index = pc.Index(self.index_name)
        logger.info(f"Initialized Pinecone index: {self.index_name}")
        
    def store(self, documents: List[Document]) -> bool:
        """
        Store documents in Pinecone.
        
        Args:
            documents: List of Document objects to store
            
        Returns:
            bool: True if storage was successful
        """
        if not self.index:
            raise RuntimeError("Index not initialized")
            
        vectors = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding")
                
            vectors.append((
                doc.id,
                doc.embedding.tolist(),
                {
                    **doc.metadata,
                    "content": doc.content  # Store content in metadata for retrieval
                }
            ))
            
        try:
            self.index.upsert(
                vectors=vectors,
                namespace=self.namespace
            )
            logger.info(f"Stored {len(documents)} documents in Pinecone")
            return True
        except Exception as e:
            logger.error(f"Error storing documents in Pinecone: {str(e)}")
            return False
            
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Query Pinecone for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_criteria: Optional filtering criteria
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.index:
            raise RuntimeError("Index not initialized")
            
        try:
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                namespace=self.namespace,
                filter=filter_criteria,
                include_metadata=True
            )
            
            documents = []
            for match in results.matches:
                metadata = dict(match.metadata)
                content = metadata.pop("content")  # Extract content from metadata
                doc = Document(
                    id=match.id,
                    content=content,
                    metadata=metadata,
                    embedding=np.array(match.values)
                )
                documents.append((doc, match.score))
                
            return documents
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            return []
            
    def delete(self, document_ids: List[str]) -> bool:
        """
        Delete documents from Pinecone.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            bool: True if deletion was successful
        """
        if not self.index:
            raise RuntimeError("Index not initialized")
            
        try:
            self.index.delete(ids=document_ids, namespace=self.namespace)
            logger.info(f"Deleted {len(document_ids)} documents from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents from Pinecone: {str(e)}")
            return False

class StorageManager:
    """
    Main class for managing document storage and retrieval.
    
    This class handles:
    1. Embedding generation using SentenceTransformers
    2. Vector database interactions
    3. Document processing and storage
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        vector_db: Optional[VectorDBInterface] = None,
        chunk_size: int = 512
    ):
        """
        Initialize the StorageManager.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            vector_db: Vector database implementation
            chunk_size: Maximum chunk size for text splitting
        """
        self.model = SentenceTransformer(model_name)
        self.vector_db = vector_db
        self.chunk_size = chunk_size
        
        if vector_db:
            # Initialize vector DB with model's embedding dimension
            vector_db.initialize(self.model.get_sentence_embedding_dimension())
            
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for the given text.
        
        Args:
            text: Input text
            
        Returns:
            numpy.ndarray: Generated embedding
        """
        return self.model.encode(text, normalize_embeddings=True)
        
    def process_and_store_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None
    ) -> str:
        """
        Process a document and store it in the vector database.
        
        Args:
            content: Document content (Markdown)
            metadata: Document metadata
            doc_id: Optional document ID
            
        Returns:
            str: Document ID
            
        Raises:
            ValueError: If content is empty or contains only whitespace
            RuntimeError: If no vector database is configured
            RuntimeError: If document storage fails
        """
        if not self.vector_db:
            raise RuntimeError("No vector database configured")
            
        # Validate content
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
            
        # Generate document ID if not provided
        if not doc_id:
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Generate embedding
        embedding = self.generate_embedding(content)
        
        # Create document object
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata,
            embedding=embedding
        )
        
        # Store in vector database
        success = self.vector_db.store([doc])
        if not success:
            raise RuntimeError(f"Failed to store document {doc_id}")
            
        return doc_id
        
    def query_similar_documents(
        self,
        query: str,
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Query for documents similar to the input query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filter_criteria: Optional filtering criteria
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        if not self.vector_db:
            raise RuntimeError("No vector database configured")
            
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Query vector database
        return self.vector_db.query(
            query_embedding,
            top_k=top_k,
            filter_criteria=filter_criteria
        )

def create_storage_manager(
    vector_db_type: str = "pinecone",
    **kwargs
) -> StorageManager:
    """
    Factory function to create a StorageManager with the specified vector database.
    
    Args:
        vector_db_type: Type of vector database to use
        **kwargs: Additional arguments for vector database initialization
        
    Returns:
        StorageManager: Configured storage manager instance
    """
    if vector_db_type == "pinecone":
        vector_db = PineconeDB(
            api_key=kwargs.get("api_key"),
            environment=kwargs.get("environment"),
            index_name=kwargs.get("index_name"),
            namespace=kwargs.get("namespace", "")
        )
    else:
        raise ValueError(f"Unsupported vector database type: {vector_db_type}")
        
    return StorageManager(
        model_name=kwargs.get("model_name", "all-MiniLM-L6-v2"),
        vector_db=vector_db,
        chunk_size=kwargs.get("chunk_size", 512)
    )

def demo():
    """
    Demonstration of the Storage & Indexing module functionality.
    """
    # Sample document
    sample_markdown = """
    # Vector Databases in RAG Systems
    
    Vector databases are essential components in modern RAG (Retrieval-Augmented Generation) systems.
    They enable efficient storage and retrieval of embeddings, making it possible to find semantically
    similar content quickly.
    
    ## Key Features
    - Fast similarity search
    - Scalable storage
    - Support for metadata filtering
    
    ## Use Cases
    1. Semantic search
    2. Content recommendation
    3. Knowledge retrieval
    """
    
    metadata = {
        "title": "Vector Databases in RAG Systems",
        "author": "John Doe",
        "date": "2024-03-15",
        "tags": ["RAG", "vector-db", "embeddings"]
    }
    
    try:
        # Initialize storage manager with Pinecone
        manager = create_storage_manager(
            vector_db_type="pinecone",
            api_key="your-api-key",  # Replace with actual key
            environment="your-environment",
            index_name="rag-demo"
        )
        
        # Store document
        doc_id = manager.process_and_store_document(
            content=sample_markdown,
            metadata=metadata
        )
        logger.info(f"Stored document with ID: {doc_id}")
        
        # Query similar documents
        query = "What are the main features of vector databases?"
        results = manager.query_similar_documents(query, top_k=3)
        
        logger.info(f"\nQuery: {query}")
        logger.info("\nResults:")
        for doc, score in results:
            logger.info(f"\nScore: {score:.3f}")
            logger.info(f"Title: {doc.metadata['title']}")
            logger.info(f"Content preview: {doc.content[:200]}...")
            
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")

if __name__ == "__main__":
    demo() 