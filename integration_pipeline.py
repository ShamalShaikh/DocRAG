"""
Integration Pipeline Module for RAG System.

This module orchestrates the data flow between different components of the RAG system:
1. Web Scraping
2. Preprocessing & Conversion
3. Storage & Indexing
4. Retrieval & QA

The pipeline ensures proper data flow and error handling between components.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache

from web_scraper import create_default_scraper, WebScraper
from html_to_markdown_converter import create_converter, HTMLToMarkdownConverter
from storage_indexing import create_storage_manager, StorageManager
from retrieval_qa import create_retrieval_qa, RetrievalQA, QAResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the integration pipeline."""
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str = "rag-demo"
    llm_model_name: str = "deepseek-r1:8b"
    chunk_size: int = 512
    max_tokens: int = 1000
    temperature: float = 0.7
    top_k: int = 3
    test_mode: bool = False  # Flag to indicate if running in test mode

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.test_mode and (self.pinecone_api_key == "test-key" or not self.pinecone_api_key):
            raise ValueError(
                "Invalid or missing Pinecone API key. "
                "Please provide a valid API key or enable test_mode for testing."
            )

@dataclass
class ProcessingResult:
    """Result of document processing through the pipeline."""
    document_id: str
    url: str
    content: str
    metadata: Dict[str, Any]
    summary: Optional[str] = None
    error: Optional[str] = None

class IntegrationPipeline:
    """
    Main class for orchestrating the RAG system pipeline.
    
    This class manages the flow of data between different modules:
    1. Fetches HTML content using the web scraper
    2. Converts HTML to Markdown with metadata
    3. Stores processed content in the vector database
    4. Provides query interface for retrieving answers
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the integration pipeline.
        
        Args:
            config: Pipeline configuration
            
        Raises:
            RuntimeError: If initialization of any component fails
            ValueError: If invalid configuration is provided
        """
        self.config = config
        
        try:
            logger.info("Initializing pipeline components...")
            
            # Initialize web scraper
            logger.info("Creating web scraper...")
            self.scraper = create_default_scraper()
            
            # Initialize HTML to Markdown converter
            logger.info("Creating HTML to Markdown converter...")
            self.converter = create_converter(
                model_name=config.llm_model_name
            )
            
            # Initialize storage manager with test mode consideration
            logger.info("Creating storage manager...")
            if config.test_mode:
                logger.info("Running in test mode - using mock storage manager")
                # In test mode, storage manager will be mocked by tests
                self.storage_manager = None
            else:
                try:
                    self.storage_manager = create_storage_manager(
                        vector_db_type="pinecone",
                        api_key=config.pinecone_api_key,
                        environment=config.pinecone_environment,
                        index_name=config.pinecone_index_name,
                        chunk_size=config.chunk_size
                    )
                except Exception as e:
                    error_msg = (
                        f"Failed to initialize Pinecone storage manager: {str(e)}. "
                        "Please ensure you have provided valid Pinecone credentials "
                        "or enable test_mode for testing."
                    )
                    logger.error(error_msg, exc_info=True)
                    raise RuntimeError(error_msg)
            
            # Initialize QA system
            logger.info("Creating QA system...")
            if config.test_mode:
                logger.info("Running in test mode - using mock QA system")
                # In test mode, QA system will be mocked by tests
                self.qa_system = None
            else:
                self.qa_system = create_retrieval_qa(
                    storage_manager=self.storage_manager,
                    model_name=config.llm_model_name,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k
                )
            
            logger.info(
                f"Pipeline initialization completed successfully "
                f"(test_mode: {config.test_mode})"
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize pipeline: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
            
    async def process_url(self, url: str) -> ProcessingResult:
        """
        Process a single URL through the complete pipeline.
        
        This method:
        1. Scrapes HTML content from the URL (async operation)
        2. Converts HTML to Markdown with metadata
        3. Stores the processed content
        
        Args:
            url: URL to process
            
        Returns:
            ProcessingResult: Result of processing including document ID and any errors
            
        Raises:
            ValueError: If URL is invalid
            RuntimeError: If processing fails at any stage
        """
        logger.info(f"Starting pipeline processing for URL: {url}")
        
        try:
            # Step 1: Fetch HTML content (async operation)
            logger.info(f"Fetching HTML content from {url}...")
            try:
                html_content = await self.scraper.scrape_html(url)
                if not html_content:
                    raise ValueError(f"No content retrieved from URL: {url}")
            except Exception as e:
                error_msg = f"Failed to scrape URL {url}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return ProcessingResult(
                    document_id="",
                    url=url,
                    content="",
                    metadata={},
                    error=error_msg
                )
            
            # Step 2: Convert HTML to Markdown
            logger.info("Converting HTML to Markdown...")
            try:
                conversion_result = self.converter.convert(html_content)
            except Exception as e:
                error_msg = f"Failed to convert HTML content: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return ProcessingResult(
                    document_id="",
                    url=url,
                    content="",
                    metadata={},
                    error=error_msg
                )
            
            # Step 3: Store in vector database
            logger.info("Storing processed content...")
            try:
                doc_id = self.storage_manager.process_and_store_document(
                    content=conversion_result["markdown"],
                    metadata={
                        "url": url,
                        "author": conversion_result["metadata"].get("author"),
                        "publication_date": conversion_result["metadata"].get("publication_date"),
                        "processed_at": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                error_msg = f"Failed to store document: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return ProcessingResult(
                    document_id="",
                    url=url,
                    content=conversion_result["markdown"],
                    metadata=conversion_result["metadata"],
                    error=error_msg
                )
            
            result = ProcessingResult(
                document_id=doc_id,
                url=url,
                content=conversion_result["markdown"],
                metadata=conversion_result["metadata"],
                summary=conversion_result.get("summary")
            )
            
            logger.info(f"Successfully processed URL: {url}")
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error processing URL {url}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ProcessingResult(
                document_id="",
                url=url,
                content="",
                metadata={},
                error=error_msg
            )
    
    async def process_urls(self, urls: List[str]) -> List[ProcessingResult]:
        """
        Process multiple URLs through the pipeline.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List[ProcessingResult]: Results for each URL
        """
        logger.info(f"Processing {len(urls)} URLs")
        results = []
        
        for url in urls:
            try:
                result = await self.process_url(url)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process URL {url}: {str(e)}")
                results.append(ProcessingResult(
                    document_id="",
                    url=url,
                    content="",
                    metadata={},
                    error=str(e)
                ))
        
        return results
    
    def query(
        self,
        query: str,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> QAResponse:
        """
        Query the RAG system to get an answer.
        
        Args:
            query: User's question
            filter_criteria: Optional filtering criteria
            
        Returns:
            QAResponse: Generated answer with source documents
            
        Raises:
            RuntimeError: If query processing fails
        """
        try:
            logger.info(f"Processing query: {query}")
            response = self.qa_system.query(
                query=query,
                filter_criteria=filter_criteria
            )
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            error_msg = f"Failed to process query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

def create_pipeline(
    pinecone_api_key: str,
    pinecone_environment: str,
    **kwargs
) -> IntegrationPipeline:
    """
    Create an integration pipeline instance.
    
    Args:
        pinecone_api_key: Pinecone API key
        pinecone_environment: Pinecone environment
        **kwargs: Additional configuration options
        
    Returns:
        IntegrationPipeline: Configured pipeline instance
    """
    config = PipelineConfig(
        pinecone_api_key=pinecone_api_key,
        pinecone_environment=pinecone_environment,
        pinecone_index_name=kwargs.get("pinecone_index_name", "rag-demo"),
        llm_model_name=kwargs.get("llm_model_name", "deepseek-r1:8b"),
        chunk_size=kwargs.get("chunk_size", 512),
        max_tokens=kwargs.get("max_tokens", 1000),
        temperature=kwargs.get("temperature", 0.7),
        top_k=kwargs.get("top_k", 3),
        test_mode=kwargs.get("test_mode", False)
    )
    
    return IntegrationPipeline(config) 