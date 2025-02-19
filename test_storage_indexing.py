"""
Test module for the Storage & Indexing functionality.

This module contains test cases for verifying the core functionality of the
storage_indexing module, including embedding generation, document storage,
and similarity search.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from storage_indexing import (
    Document,
    VectorDBInterface,
    StorageManager,
    create_storage_manager
)

# Sample test data
SAMPLE_MARKDOWN = """
# Test Document

This is a test document for verifying the storage and indexing functionality.
It contains multiple paragraphs and some basic Markdown formatting.

## Section 1
- Point 1
- Point 2

## Section 2
1. First item
2. Second item
"""

SAMPLE_METADATA = {
    "title": "Test Document",
    "author": "Test Author",
    "date": "2024-03-15",
    "tags": ["test", "document"]
}

class MockVectorDB(VectorDBInterface):
    """Mock vector database implementation for testing."""
    
    def __init__(self):
        self.documents = {}
        self.initialized = False
        self.dimension = None
        
    def initialize(self, dimension: int) -> None:
        self.initialized = True
        self.dimension = dimension
        
    def store(self, documents: list[Document]) -> bool:
        try:
            for doc in documents:
                self.documents[doc.id] = doc
            return True
        except Exception:
            return False
            
    def query(self, query_embedding, top_k=5, filter_criteria=None):
        # Simple mock implementation returning stored documents
        results = []
        for doc_id, doc in self.documents.items():
            # Calculate mock similarity score (random for testing)
            score = np.random.random()
            results.append((doc, score))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
        
    def delete(self, document_ids: list[str]) -> bool:
        try:
            for doc_id in document_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
            return True
        except Exception:
            return False

@pytest.fixture
def mock_vector_db():
    """Fixture providing a mock vector database."""
    return MockVectorDB()

@pytest.fixture
def storage_manager(mock_vector_db):
    """Fixture providing a StorageManager instance with mock vector DB."""
    return StorageManager(vector_db=mock_vector_db)

def test_document_creation():
    """Test Document dataclass creation and attributes."""
    doc = Document(
        id="test_doc",
        content="Test content",
        metadata={"key": "value"},
        embedding=np.array([0.1, 0.2, 0.3])
    )
    
    assert doc.id == "test_doc"
    assert doc.content == "Test content"
    assert doc.metadata == {"key": "value"}
    assert np.array_equal(doc.embedding, np.array([0.1, 0.2, 0.3]))

def test_embedding_generation(storage_manager):
    """Test embedding generation from text."""
    embedding = storage_manager.generate_embedding(SAMPLE_MARKDOWN)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1  # Should be a 1D vector
    assert embedding.shape[0] > 0  # Should have non-zero dimension

def test_document_storage(storage_manager):
    """Test document storage functionality."""
    doc_id = storage_manager.process_and_store_document(
        content=SAMPLE_MARKDOWN,
        metadata=SAMPLE_METADATA
    )
    
    assert doc_id is not None
    assert doc_id in storage_manager.vector_db.documents
    
    stored_doc = storage_manager.vector_db.documents[doc_id]
    assert stored_doc.content == SAMPLE_MARKDOWN
    assert stored_doc.metadata == SAMPLE_METADATA
    assert stored_doc.embedding is not None

def test_document_query(storage_manager):
    """Test document querying functionality."""
    # Store a document first
    doc_id = storage_manager.process_and_store_document(
        content=SAMPLE_MARKDOWN,
        metadata=SAMPLE_METADATA
    )
    
    # Query for similar documents
    query = "Test document with Markdown"
    results = storage_manager.query_similar_documents(query, top_k=3)
    
    assert len(results) > 0
    assert all(isinstance(doc, Document) for doc, score in results)
    assert all(isinstance(score, float) for doc, score in results)
    assert all(0 <= score <= 1 for doc, score in results)

def test_storage_manager_creation():
    """Test StorageManager factory function."""
    with pytest.raises(ValueError):
        # Should raise error for unsupported DB type
        create_storage_manager(vector_db_type="unsupported")
        
    with patch("storage_indexing.PineconeDB") as mock_pinecone:
        manager = create_storage_manager(
            vector_db_type="pinecone",
            api_key="test",
            environment="test",
            index_name="test"
        )
        assert isinstance(manager, StorageManager)
        mock_pinecone.assert_called_once()

def test_error_handling(storage_manager):
    """Test error handling in storage operations."""
    # Test with empty content
    with pytest.raises(ValueError):
        storage_manager.process_and_store_document("", {})
        
    # Test with missing vector DB
    manager_no_db = StorageManager(vector_db=None)
    with pytest.raises(RuntimeError):
        manager_no_db.process_and_store_document(SAMPLE_MARKDOWN, SAMPLE_METADATA)

def test_metadata_handling(storage_manager):
    """Test handling of different metadata types."""
    metadata_with_dates = {
        **SAMPLE_METADATA,
        "created_at": datetime.now(),
        "nested": {"key": "value"},
        "numbers": [1, 2, 3]
    }
    
    doc_id = storage_manager.process_and_store_document(
        content=SAMPLE_MARKDOWN,
        metadata=metadata_with_dates
    )
    
    stored_doc = storage_manager.vector_db.documents[doc_id]
    assert stored_doc.metadata == metadata_with_dates

if __name__ == "__main__":
    pytest.main([__file__]) 