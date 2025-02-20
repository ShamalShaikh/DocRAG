"""
FastAPI Server for RAG System.

This module provides REST API endpoints for web scraping, document conversion,
and question answering functionality. It integrates with the web scraping,
storage & indexing, and retrieval & QA modules.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from functools import lru_cache
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

from storage_indexing import create_storage_manager
from retrieval_qa import create_retrieval_qa
from web_scraper import create_default_scraper
from html_to_markdown_converter import create_converter

# Configure logging with more detailed format and file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# Add file handler for persistent logging
file_handler = logging.FileHandler('api_server.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

class Settings(BaseSettings):
    """Configuration settings for the API server."""
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str = "rag-demo"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    cors_origins: List[str] = ["http://localhost:8501", "http://localhost:3000", "http://127.0.0.1:8501"]
    debug_mode: bool = False
    log_level: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

# Create settings instance at module level
settings = Settings()

def initialize_components(settings: Settings):
    """
    Initialize system components with proper error handling.
    
    Args:
        settings: Application settings
        
    Returns:
        Tuple[StorageManager, RetrievalQA, WebScraper, HTMLToMarkdownConverter]: Initialized components
        
    Raises:
        RuntimeError: If component initialization fails
    """
    components = {}
    try:
        # Initialize storage manager
        logger.info("Initializing storage manager with Pinecone...")
        storage_manager = create_storage_manager(
            vector_db_type="pinecone",
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment,
            index_name=settings.pinecone_index_name,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region
        )
        components["storage_manager"] = storage_manager
        logger.info("Storage manager initialized successfully")
        
        # Initialize QA system
        logger.info("Initializing QA system...")
        qa_system = create_retrieval_qa(storage_manager)
        components["qa_system"] = qa_system
        logger.info("QA system initialized successfully")
        
        # Initialize web scraper
        logger.info("Initializing web scraper...")
        scraper = create_default_scraper()
        components["scraper"] = scraper
        logger.info("Web scraper initialized successfully")

        # Initialize HTML to Markdown converter
        logger.info("Initializing HTML to Markdown converter...")
        converter = create_converter()
        components["converter"] = converter
        logger.info("HTML to Markdown converter initialized successfully")
        
        return storage_manager, qa_system, scraper, converter
        
    except Exception as e:
        # Log which component failed to initialize
        failed_component = next(
            (name for name, comp in components.items() if comp is None),
            "unknown"
        )
        error_msg = f"Failed to initialize {failed_component}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="API for web scraping, document processing, and question answering",
    version="1.0.0"
)

# Add CORS middleware immediately after creating the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    error_msg = f"Unhandled error: {str(exc)}"
    logger.error(error_msg, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": error_msg}
    )

# Replace on_event handlers with lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    try:
        # Startup: Configure logging and initialize components
        logger.setLevel(getattr(logging, settings.log_level.upper()))
        logger.info("Starting application initialization...")
        global storage_manager, qa_system, scraper, converter
        storage_manager, qa_system, scraper, converter = initialize_components(settings)
        logger.info("Application initialization completed successfully")
        yield
    except Exception as e:
        error_msg = f"Failed to start application: {str(e)}"
        logger.critical(error_msg, exc_info=True)
        raise RuntimeError(error_msg)
    finally:
        # Shutdown: Cleanup resources
        logger.info("Shutting down application...")
        # Add cleanup code here if needed

# Update FastAPI app to use the lifespan context manager
app = FastAPI(
    title="RAG System API",
    description="API for web scraping, document processing, and question answering",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response Models
class ScrapingTarget(BaseModel):
    """Model for scraping target configuration."""
    url: HttpUrl
    schedule: Optional[str] = None  # Cron expression
    metadata: Optional[Dict[str, Any]] = None

class ScrapingStatus(BaseModel):
    """Model for scraping job status."""
    job_id: str
    url: HttpUrl
    status: str
    last_run: datetime
    next_run: Optional[datetime] = None
    error: Optional[str] = None

class QueryRequest(BaseModel):
    """Model for query request."""
    query: str
    filter_criteria: Optional[Dict[str, Any]] = None
    max_results: Optional[int] = 3

class QueryResponse(BaseModel):
    """Model for query response."""
    answer: str
    sources: List[Dict[str, Any]]
    total_tokens: int

class AnnotationRequest(BaseModel):
    """Model for document annotation."""
    document_id: str
    annotation: str
    highlight_text: str
    metadata: Optional[Dict[str, Any]] = None

class AnnotationResponse(BaseModel):
    """Model for annotation response."""
    status: str
    message: str
    annotation_id: Optional[str] = None

# API Endpoints
async def process_and_store_url(url: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Process a URL through the complete pipeline: scrape → convert → store.
    
    Args:
        url: URL to process
        metadata: Optional metadata to store with the document
        
    Returns:
        str: Document ID of the stored document
        
    Raises:
        HTTPException: If any step in the pipeline fails
    """
    try:
        # Step 1: Scrape HTML content
        logger.info(f"Scraping content from URL: {url}")
        html_content = await scraper.scrape_html(url)
        if not html_content:
            raise ValueError(f"No content retrieved from URL: {url}")
            
        # Step 2: Convert HTML to Markdown
        logger.info("Converting HTML to Markdown")
        conversion_result = converter.convert(html_content)
        
        # Step 3: Prepare metadata
        doc_metadata = {
            "url": url,
            "processed_at": datetime.now().isoformat(),
            **(metadata or {}),  # Include any provided metadata
            **(conversion_result.get("metadata", {}))  # Include metadata from conversion
        }
        
        # Remove any None values from metadata
        doc_metadata = {k: v for k, v in doc_metadata.items() if v is not None}
        
        # Step 4: Store in vector database
        logger.info("Storing document in vector database")
        doc_id = storage_manager.process_and_store_document(
            content=conversion_result["markdown"],
            metadata=doc_metadata
        )
        
        logger.info(f"Successfully stored document with ID: {doc_id}")
        return doc_id
        
    except Exception as e:
        error_msg = f"Failed to process URL {url}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

@app.post("/api/scrape", response_model=ScrapingStatus)
async def start_scraping(
    target: ScrapingTarget
):
    """
    Start a web scraping job for the specified target and store the processed content.
    
    This endpoint:
    1. Initiates scraping of the target URL
    2. Processes the HTML content to Markdown
    3. Stores the document in the vector database
    4. Optionally schedules recurring scraping
    
    Args:
        target: Scraping target configuration
        
    Returns:
        ScrapingStatus: Status of the created scraping job
        
    Raises:
        HTTPException: If scraping or processing fails
    """
    try:
        logger.info(f"Starting scraping job for URL: {target.url}")
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process and store the document immediately
        doc_id = await process_and_store_url(
            url=str(target.url),
            metadata=target.metadata
        )
        
        # Schedule future scraping if requested
        next_run = None
        if target.schedule:
            logger.info(f"Scheduling scraping job {job_id} with schedule: {target.schedule}")
            scraper.schedule_scraping(
                task=scraper.scrape_html,
                interval=3600,  # Convert schedule to interval
                url=str(target.url)
            )
            next_run = datetime.now()
        
        status = ScrapingStatus(
            job_id=job_id,
            url=target.url,
            status="scheduled" if target.schedule else "completed",
            last_run=datetime.now(),
            next_run=next_run
        )
        
        logger.info(f"Created scraping job: {status.dict()}, Document ID: {doc_id}")
        return status
        
    except Exception as e:
        error_msg = f"Scraping failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

@app.get("/api/scrape/{job_id}", response_model=ScrapingStatus)
async def get_scraping_status(job_id: str):
    """
    Get the status of a scraping job.
    
    Args:
        job_id: ID of the scraping job
        
    Returns:
        ScrapingStatus: Current status of the scraping job
    """
    logger.info(f"Fetching status for scraping job: {job_id}")
    
    # Stub implementation
    status = ScrapingStatus(
        job_id=job_id,
        url="http://example.com",  # Placeholder URL
        status="in_development",
        last_run=datetime.now(),
        next_run=None,
        error="This endpoint is under development. Job status tracking will be implemented in a future update."
    )
    logger.info(f"Returning stub status for job {job_id}: {status.dict()}")
    return status

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query using the RAG pipeline.
    
    Args:
        request: Query request containing the question and optional filters
        
    Returns:
        QueryResponse: Generated answer with source documents
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        logger.info(f"Processing query: {request.query}")
        response = qa_system.query(
            query=request.query,
            filter_criteria=request.filter_criteria
        )
        
        result = QueryResponse(
            answer=response.answer,
            sources=[{
                "content": source.content,
                "metadata": source.metadata,
                "similarity_score": source.similarity_score
            } for source in response.sources],
            total_tokens=response.total_tokens
        )
        
        logger.info(f"Query processed successfully. Total tokens: {response.total_tokens}")
        return result
        
    except Exception as e:
        error_msg = f"Query failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

@app.post("/api/annotate", response_model=AnnotationResponse)
async def annotate_document(request: AnnotationRequest):
    """
    Add an annotation to a document.
    
    Args:
        request: Annotation request containing document ID and annotation text
        
    Returns:
        AnnotationResponse: Status of the annotation operation
    """
    logger.info(f"Received annotation request for document: {request.document_id}")
    
    # Stub implementation
    annotation_id = f"annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    response = AnnotationResponse(
        status="in_development",
        message="Document annotation feature is under development. This is a stub response.",
        annotation_id=annotation_id
    )
    logger.info(f"Returning stub response for annotation request: {response.dict()}")
    return response

@app.get("/api/stats")
async def get_system_stats():
    """
    Get system statistics (document count, embedding stats, etc.).
    
    Returns:
        Dict: System statistics
        
    Raises:
        HTTPException: If stats collection fails
    """
    try:
        logger.info("Collecting system statistics")
        # TODO: Implement actual stats collection
        stats = {
            "total_documents": 0,
            "total_embeddings": 0,
            "last_update": datetime.now().isoformat(),
            "status": "Statistics collection is under development"
        }
        logger.info("Statistics collected successfully")
        return stats
        
    except Exception as e:
        error_msg = f"Failed to get stats: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=settings.debug_mode  # Changed debug to reload
    ) 