"""
Integration Tests for the RAG System Pipeline.

This module contains tests that verify the end-to-end functionality of the RAG system,
including data flow between components and error handling.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

from integration_pipeline import (
    create_pipeline,
    PipelineConfig,
    ProcessingResult,
    IntegrationPipeline
)

# Sample test data
SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta name="author" content="Test Author">
    <meta name="publication_date" content="2024-03-15">
    <title>Test Article</title>
</head>
<body>
    <article>
        <h1>Vector Databases in RAG Systems</h1>
        <p>Vector databases are essential components in modern RAG systems.
        They enable efficient storage and retrieval of embeddings.</p>
        <h2>Key Features</h2>
        <ul>
            <li>Fast similarity search</li>
            <li>Scalable storage</li>
            <li>Metadata filtering</li>
        </ul>
    </article>
</body>
</html>
"""

EXPECTED_MARKDOWN = """
# Vector Databases in RAG Systems

Vector databases are essential components in modern RAG systems.
They enable efficient storage and retrieval of embeddings.

## Key Features

- Fast similarity search
- Scalable storage
- Metadata filtering
"""

@pytest.fixture
def mock_pinecone():
    """
    Fixture providing a mock Pinecone client.
    
    This fixture mocks the entire Pinecone client and its essential methods
    to prevent any real API calls during testing.
    """
    with patch("pinecone.Pinecone") as mock_pc:
        # Create a mock index instance
        mock_index = MagicMock()
        mock_index.upsert.return_value = None
        mock_index.query.return_value = MagicMock(
            matches=[
                MagicMock(
                    id="test_doc_id",
                    score=0.95,
                    metadata={
                        "content": EXPECTED_MARKDOWN,
                        "title": "Test Article"
                    }
                )
            ]
        )
        
        # Configure the Pinecone client mock
        mock_pc.return_value.list_indexes.return_value = ["test-index"]
        mock_pc.return_value.Index.return_value = mock_index
        
        # Mock the ServerlessSpec for index creation
        mock_pc.ServerlessSpec = MagicMock()
        
        yield mock_pc

@pytest.fixture
def mock_components(mock_pinecone):
    """
    Fixture providing mock components for testing.
    
    This fixture mocks all external dependencies and configures them with
    appropriate return values for testing. Uses AsyncMock for async methods.
    """
    with patch("web_scraper.create_default_scraper") as mock_scraper, \
         patch("html_to_markdown_converter.create_converter") as mock_converter, \
         patch("storage_indexing.create_storage_manager") as mock_storage, \
         patch("retrieval_qa.create_retrieval_qa") as mock_qa:
        
        # Configure mock scraper with AsyncMock for async methods
        scraper = Mock()
        scraper.scrape_html = AsyncMock(return_value=SAMPLE_HTML)
        mock_scraper.return_value = scraper
        
        # Configure mock converter
        converter = Mock()
        converter.convert.return_value = {
            "markdown": EXPECTED_MARKDOWN,
            "metadata": {
                "author": "Test Author",
                "publication_date": "2024-03-15",
                "title": "Test Article"
            },
            "summary": "An article about vector databases in RAG systems."
        }
        mock_converter.return_value = converter
        
        # Configure mock storage manager with Pinecone integration
        storage = Mock()
        storage.process_and_store_document.return_value = "test_doc_id"
        storage.vector_db = mock_pinecone.return_value.Index.return_value
        mock_storage.return_value = storage
        
        # Configure mock QA system
        qa = Mock()
        qa.query.return_value = Mock(
            answer="Vector databases provide fast similarity search and scalable storage.",
            sources=[Mock(
                content=EXPECTED_MARKDOWN,
                metadata={"title": "Test Article"},
                similarity_score=0.95
            )],
            total_tokens=150
        )
        mock_qa.return_value = qa
        
        yield {
            "scraper": scraper,
            "converter": converter,
            "storage": storage,
            "qa": qa,
            "pinecone": mock_pinecone
        }

@pytest.fixture
def pipeline(mock_components):
    """
    Fixture providing a configured pipeline instance for testing.
    
    Returns a pipeline instance in test mode with mocked components.
    """
    pipeline = create_pipeline(
        pinecone_api_key="test-key",
        pinecone_environment="test-env",
        pinecone_index_name="test-index",
        test_mode=True  # Enable test mode
    )
    
    # Assign the mock components
    pipeline.scraper = mock_components["scraper"]
    pipeline.converter = mock_components["converter"]
    pipeline.storage_manager = mock_components["storage"]
    pipeline.qa_system = mock_components["qa"]
    
    return pipeline

def test_pipeline_config_validation():
    """Test pipeline configuration validation."""
    # Test with invalid API key in production mode
    with pytest.raises(ValueError) as exc_info:
        PipelineConfig(
            pinecone_api_key="test-key",
            pinecone_environment="test-env",
            test_mode=False
        )
    assert "Invalid or missing Pinecone API key" in str(exc_info.value)
    
    # Test with valid configuration in test mode
    config = PipelineConfig(
        pinecone_api_key="test-key",
        pinecone_environment="test-env",
        test_mode=True
    )
    assert config.test_mode is True

def test_pipeline_initialization_with_invalid_key():
    """Test pipeline initialization with invalid API key."""
    with pytest.raises(ValueError) as exc_info:
        create_pipeline(
            pinecone_api_key="test-key",
            pinecone_environment="test-env",
            test_mode=False  # Disable test mode
        )
    assert "Invalid or missing Pinecone API key" in str(exc_info.value)

def test_pipeline_initialization_in_test_mode(mock_components):
    """Test pipeline initialization in test mode."""
    pipeline = create_pipeline(
        pinecone_api_key="test-key",
        pinecone_environment="test-env",
        test_mode=True
    )
    assert pipeline.config.test_mode is True
    # Components should be None initially in test mode
    assert pipeline.storage_manager is None
    assert pipeline.qa_system is None

@pytest.mark.asyncio
async def test_process_single_url(pipeline, mock_components):
    """
    Test processing a single URL through the pipeline.
    
    This test verifies that:
    1. The scraper's scrape_html method is called with the correct URL
    2. The HTML content is properly awaited
    3. The converter's convert method is called with the scraped HTML
    4. The storage manager processes and stores the document
    5. The result contains the expected data
    """
    url = "https://example.com/article"
    result = await pipeline.process_url(url)
    
    # Verify component interactions
    mock_components["scraper"].scrape_html.assert_awaited_once_with(url)
    mock_components["converter"].convert.assert_called_once_with(SAMPLE_HTML)
    mock_components["storage"].process_and_store_document.assert_called_once()
    
    # Verify result structure and content
    assert isinstance(result, ProcessingResult)
    assert result.document_id == "test_doc_id"
    assert result.url == url
    assert result.content == EXPECTED_MARKDOWN
    assert result.metadata["author"] == "Test Author"
    assert result.error is None

@pytest.mark.asyncio
async def test_process_multiple_urls(pipeline, mock_components):
    """Test processing multiple URLs through the pipeline."""
    urls = [
        "https://example.com/article1",
        "https://example.com/article2"
    ]
    
    results = await pipeline.process_urls(urls)
    
    # Verify results
    assert len(results) == 2
    assert all(isinstance(r, ProcessingResult) for r in results)
    assert all(r.error is None for r in results)
    assert mock_components["scraper"].scrape_html.await_count == 2

@pytest.mark.asyncio
async def test_error_handling_invalid_url(pipeline, mock_components):
    """Test error handling when processing an invalid URL."""
    # Configure AsyncMock to raise an error
    mock_components["scraper"].scrape_html.side_effect = ValueError("Invalid URL")
    
    result = await pipeline.process_url("invalid-url")
    
    # Verify error handling
    assert isinstance(result, ProcessingResult)
    assert result.error is not None
    assert "Invalid URL" in result.error
    assert result.document_id == ""
    
    # Verify the converter was not called due to the error
    mock_components["converter"].convert.assert_not_called()

@pytest.mark.asyncio
async def test_scraper_network_error(pipeline, mock_components):
    """Test handling of network errors during scraping."""
    # Simulate a network error
    mock_components["scraper"].scrape_html.side_effect = ConnectionError("Network error")
    
    result = await pipeline.process_url("https://example.com")
    
    assert result.error is not None
    assert "Network error" in result.error
    mock_components["converter"].convert.assert_not_called()

def test_query_processing(pipeline, mock_components):
    """Test query processing through the RAG system."""
    query = "What are the key features of vector databases?"
    response = pipeline.query(query)
    
    # Verify QA system interaction
    mock_components["qa"].query.assert_called_once_with(
        query=query,
        filter_criteria=None
    )
    
    # Verify response
    assert response.answer == "Vector databases provide fast similarity search and scalable storage."
    assert len(response.sources) == 1
    assert response.total_tokens == 150

@pytest.mark.asyncio
async def test_component_failure_handling(pipeline, mock_components):
    """Test handling of component failures during processing."""
    # Simulate converter failure
    mock_components["converter"].convert.side_effect = RuntimeError("Conversion failed")
    
    result = await pipeline.process_url("https://example.com/article")
    
    # Verify error handling
    assert isinstance(result, ProcessingResult)
    assert result.error is not None
    assert "Conversion failed" in result.error
    assert result.document_id == ""

def test_query_with_filter_criteria(pipeline, mock_components):
    """Test query processing with filter criteria."""
    query = "What are vector databases?"
    filter_criteria = {"domain": "tech"}
    
    pipeline.query(query, filter_criteria=filter_criteria)
    
    # Verify QA system interaction with filters
    mock_components["qa"].query.assert_called_once_with(
        query=query,
        filter_criteria=filter_criteria
    )

def test_pinecone_integration(mock_components):
    """Test Pinecone integration in test mode."""
    # Test index creation
    mock_pinecone = mock_components["pinecone"]
    mock_pinecone.assert_not_called()
    
    # Verify storage operations
    storage = mock_components["storage"]
    storage.process_and_store_document("test_doc_id")
    storage.vector_db.upsert.assert_not_called()  # Should not make real API calls

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 