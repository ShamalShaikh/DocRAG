"""
Integration Tests with Real Components for RAG System.

This module contains integration tests that use real components
to verify actual data transformations and storage operations. These tests
complement the mock-based tests by validating real-world behavior.
"""

import os
import pytest
import pytest_asyncio
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from integration_pipeline import (
    create_pipeline,
    PipelineConfig,
    ProcessingResult,
    IntegrationPipeline
)
from html_to_markdown_converter import create_converter
from storage_indexing import create_storage_manager
from web_scraper import create_default_scraper

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Test URLs
TEST_URLS = [
    "https://shamalshaikh.github.io/ShamalBlog/posts/FastAPI-LLM-Integration/",
    "https://shamalshaikh.github.io/ShamalBlog/posts/AI-Weekly-Digest/"
]

class TestConfig:
    """Configuration for real component tests."""
    def __init__(self):
        # Load environment variables
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_index = os.getenv("PINECONE_INDEX_NAME", "rag-test-index")
        
        if not self.pinecone_api_key or not self.pinecone_env:
            logger.warning("Pinecone credentials not found in environment variables")
            
        self.use_real_pinecone = bool(self.pinecone_api_key and self.pinecone_env)
        
        # Log configuration for debugging
        logger.info(f"Test configuration initialized:")
        logger.info(f"  Using real Pinecone: {self.use_real_pinecone}")
        logger.info(f"  Pinecone index: {self.pinecone_index}")

@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return TestConfig()

@pytest.fixture
def real_converter():
    """
    Fixture providing a real HTML-to-Markdown converter instance.
    
    This uses the actual converter implementation rather than a mock.
    """
    return create_converter(model_name="reader-lm:1.5b")

@pytest.fixture
def real_scraper():
    """
    Fixture providing a real web scraper instance.
    """
    return create_default_scraper()

@pytest.fixture
def real_storage_manager(test_config):
    """
    Fixture providing a real storage manager with vector DB.
    
    If Pinecone credentials are available, uses real Pinecone;
    otherwise, skips tests requiring vector storage.
    """
    if not test_config.use_real_pinecone:
        pytest.skip("Pinecone credentials not available")
        
    try:
        storage_manager = create_storage_manager(
            vector_db_type="pinecone",
            api_key=test_config.pinecone_api_key,
            environment=test_config.pinecone_env,
            index_name=test_config.pinecone_index
        )
        logger.info("Successfully created real storage manager")
        return storage_manager
    except Exception as e:
        logger.error(f"Failed to create storage manager: {str(e)}")
        pytest.skip(f"Failed to initialize storage manager: {str(e)}")

@pytest_asyncio.fixture
async def pipeline_with_real_converter(real_converter, real_scraper, real_storage_manager, test_config):
    """
    Fixture providing a pipeline with real components (converter, scraper, and storage).
    
    This fixture now includes the real storage manager to ensure proper end-to-end testing.
    If Pinecone credentials are not available, the test will be skipped.
    """
    if not test_config.use_real_pinecone:
        pytest.skip("Pinecone credentials not available")
        
    pipeline = create_pipeline(
        pinecone_api_key=test_config.pinecone_api_key,
        pinecone_environment=test_config.pinecone_env,
        pinecone_index_name=test_config.pinecone_index,
        test_mode=False  # Using real components, so test_mode should be False
    )
    
    # Use real components
    pipeline.converter = real_converter
    pipeline.scraper = real_scraper
    pipeline.storage_manager = real_storage_manager
    
    logger.info("Created pipeline with real components")
    return pipeline

@pytest.mark.asyncio
async def test_real_html_conversion(pipeline_with_real_converter):
    """
    Test the real HTML-to-Markdown conversion process with actual web scraping
    and document storage.
    
    This test verifies that:
    1. The scraper can fetch real HTML content
    2. The converter can handle real HTML input
    3. The conversion preserves document structure
    4. Metadata is correctly extracted
    5. The document is successfully stored in the vector database
    """
    # Get the pipeline instance
    pipeline = pipeline_with_real_converter
    
    # Verify pipeline components
    assert pipeline.storage_manager is not None, "Storage manager should be initialized"
    assert pipeline.converter is not None, "Converter should be initialized"
    assert pipeline.scraper is not None, "Scraper should be initialized"
    
    url = TEST_URLS[0]
    logger.info(f"Testing URL: {url}")
    
    # Process the URL
    result = await pipeline.process_url(url)
    
    # Verify successful processing
    assert result.error is None, f"Processing failed: {result.error}"
    assert result.content, "No content was generated"
    assert result.document_id, "No document ID was generated (storage may have failed)"
    
    # Verify document structure is preserved
    markdown = result.content.lower()
    assert "# " in markdown, "Main heading not found"
    assert "## " in markdown, "Section heading not found"
    
    # Verify metadata extraction
    assert result.metadata is not None, "No metadata extracted"
    assert "author" in result.metadata, "Author not extracted"
    assert "publication_date" in result.metadata, "Publication date not extracted"
    
    # Log success
    logger.info(f"Successfully processed and stored document with ID: {result.document_id}")

@pytest.mark.asyncio
async def test_real_vector_storage(pipeline_with_real_converter):
    """
    Test the complete pipeline with real components.
    
    This test verifies that:
    1. Real HTML can be scraped and processed
    2. Documents can be stored in the vector DB
    3. Metadata is properly indexed
    4. Similar documents can be retrieved
    """
    # Get the pipeline instance
    pipeline = pipeline_with_real_converter
    
    # Process and store a document
    url = TEST_URLS[0]
    result = await pipeline.process_url(url)
    
    assert result.error is None, f"Processing failed: {result.error}"
    assert result.document_id, "No document ID was generated"
    
    # Test retrieval via QA
    query = "What are the key features discussed in this article?"
    qa_response = pipeline.query(query)  # query is not async
    
    assert qa_response.answer, "No answer was generated"
    assert qa_response.sources, "No sources were retrieved"
    assert len(qa_response.sources) > 0, "No relevant sources found"
    
    # Log the answer for inspection
    logger.info(f"Generated answer: {qa_response.answer}")

@pytest.mark.asyncio
async def test_real_metadata_filtering(pipeline_with_real_converter):
    """
    Test metadata filtering with real components.
    
    This test verifies that:
    1. Multiple documents can be processed
    2. Documents can be filtered by metadata
    3. Search results respect metadata constraints
    """
    # Get the pipeline instance
    pipeline = pipeline_with_real_converter
    
    # Process multiple documents
    results = []
    for url in TEST_URLS:
        result = await pipeline.process_url(url)
        assert result.error is None, f"Processing failed for {url}: {result.error}"
        results.append(result)
    
    # Verify we have results
    assert len(results) > 0, "No documents were processed"
    
    # Extract an author from the first result for filtering
    author = results[0].metadata.get("author")
    if author:
        # Test querying with metadata filter
        filter_criteria = {"author": author}
        query = "What topics are covered in these articles?"
        
        qa_response = pipeline.query(  # query is not async
            query,
            filter_criteria=filter_criteria
        )
        
        assert qa_response.sources, "No sources found with metadata filter"
        for source in qa_response.sources:
            assert source.metadata.get("author") == author, \
                "Metadata filtering not working correctly"

# def test_direct_converter_usage(real_converter, real_scraper):
#     """
#     Test the HTML-to-Markdown converter directly with real HTML content.
    
#     This test verifies the converter's behavior without pipeline integration.
#     """
#     # First get real HTML content
#     html_content = real_scraper._session.get(TEST_URLS[0]).text
    
#     # Convert the HTML
#     result = real_converter.convert(html_content)
    
#     assert "markdown" in result, "No markdown content generated"
#     assert "metadata" in result, "No metadata extracted"
    
#     markdown = result["markdown"].lower()
#     assert "#" in markdown, "Headings not converted"
    
#     logger.info("Conversion result preview:")
#     logger.info("\n".join(markdown.split("\n")[:5]))

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 