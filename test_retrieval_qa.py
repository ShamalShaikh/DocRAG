"""
Test module for the Retrieval & QA functionality.

This module contains test cases for verifying the core functionality of the
retrieval_qa module, including document retrieval and answer generation.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Tuple

import numpy as np
from storage_indexing import Document, StorageManager
from retrieval_qa import (
    RetrievalQA,
    SearchResult,
    QAResponse,
    create_retrieval_qa
)

# Sample test data
SAMPLE_DOCUMENTS = [
    (Document(
        id="doc1",
        content="Vector databases are essential for efficient similarity search.",
        metadata={"title": "Vector DBs", "author": "Test Author"},
        embedding=np.array([0.1, 0.2, 0.3])
    ), 0.95),
    (Document(
        id="doc2",
        content="RAG systems combine retrieval with generation for better answers.",
        metadata={"title": "RAG Systems", "author": "Test Author"},
        embedding=np.array([0.4, 0.5, 0.6])
    ), 0.85)
]

SAMPLE_LLM_RESPONSE = Mock(
    response="Vector databases enable efficient similarity search in RAG systems.",
    prompt_eval_count=100,
    eval_count=50
)

@pytest.fixture
def mock_storage_manager():
    """Fixture providing a mock storage manager."""
    manager = Mock(spec=StorageManager)
    manager.query_similar_documents.return_value = SAMPLE_DOCUMENTS
    return manager

@pytest.fixture
def mock_llm():
    """Fixture providing a mock LLM client."""
    llm = Mock()
    llm.generate.return_value = SAMPLE_LLM_RESPONSE
    return llm

@pytest.fixture
def retrieval_qa(mock_storage_manager, mock_llm):
    """Fixture providing a RetrievalQA instance with mocked dependencies."""
    with patch('retrieval_qa.ollama.Client', return_value=mock_llm):
        return create_retrieval_qa(mock_storage_manager)

def test_document_retrieval(retrieval_qa, mock_storage_manager):
    """Test document retrieval functionality."""
    query = "How do vector databases work?"
    results = retrieval_qa.retrieve_documents(query)
    
    # Verify storage manager was called correctly
    mock_storage_manager.query_similar_documents.assert_called_once_with(
        query=query,
        top_k=3,
        filter_criteria=None
    )
    
    # Verify results
    assert len(results) == len(SAMPLE_DOCUMENTS)
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].content == SAMPLE_DOCUMENTS[0][0].content
    assert results[0].similarity_score == SAMPLE_DOCUMENTS[0][1]

def test_prompt_construction(retrieval_qa):
    """Test prompt construction from query and documents."""
    query = "Test query"
    documents = [
        SearchResult(
            content="Test content",
            metadata={"title": "Test Doc"},
            similarity_score=0.9
        )
    ]
    
    prompt = retrieval_qa._construct_prompt(query, documents)
    
    # Verify prompt structure
    assert "You are a helpful AI assistant" in prompt
    assert "Test content" in prompt
    assert "Test Doc" in prompt
    assert f"Question: {query}" in prompt
    assert prompt.endswith("Answer: ")

def test_answer_generation(retrieval_qa, mock_llm):
    """Test answer generation using LLM."""
    query = "Test query"
    documents = [
        SearchResult(
            content="Test content",
            metadata={"title": "Test Doc"},
            similarity_score=0.9
        )
    ]
    
    response = retrieval_qa.generate_answer(query, documents)
    
    # Verify LLM was called
    mock_llm.generate.assert_called_once()
    
    # Verify response structure
    assert isinstance(response, QAResponse)
    assert response.answer == SAMPLE_LLM_RESPONSE.response
    assert response.sources == documents
    assert response.total_tokens == (
        SAMPLE_LLM_RESPONSE.prompt_eval_count +
        SAMPLE_LLM_RESPONSE.eval_count
    )

def test_complete_query_pipeline(retrieval_qa):
    """Test the complete query pipeline from retrieval to answer generation."""
    query = "How do vector databases work?"
    response = retrieval_qa.query(query)
    
    # Verify response structure
    assert isinstance(response, QAResponse)
    assert response.answer == SAMPLE_LLM_RESPONSE.response
    assert len(response.sources) == len(SAMPLE_DOCUMENTS)
    assert response.total_tokens > 0

def test_empty_results_handling(retrieval_qa, mock_storage_manager):
    """Test handling of queries with no relevant documents."""
    # Configure mock to return no results
    mock_storage_manager.query_similar_documents.return_value = []
    
    response = retrieval_qa.query("Query with no results")
    
    assert "couldn't find any relevant information" in response.answer
    assert len(response.sources) == 0
    assert response.total_tokens == 0

def test_error_handling(retrieval_qa, mock_llm):
    """Test error handling in the QA pipeline."""
    # Configure mock to raise an exception
    mock_llm.generate.side_effect = Exception("LLM error")
    
    with pytest.raises(Exception):
        retrieval_qa.query("Test query")

if __name__ == "__main__":
    pytest.main([__file__]) 