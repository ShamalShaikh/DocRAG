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

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

from storage_indexing import create_storage_manager
from retrieval_qa import create_retrieval_qa
from web_scraper import create_default_scraper

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
    cors_origins: List[str] = ["*"]
    debug_mode: bool = False
    log_level: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings loaded from environment variables
        
    Raises:
        ValueError: If required environment variables are missing
    """
    try:
        settings = Settings()
        logger.info("Successfully loaded configuration settings")
        return settings
    except Exception as e:
        error_msg = f"Failed to load configuration settings: {str(e)}"
        logger.critical(error_msg, exc_info=True)
        raise ValueError(error_msg)

def initialize_components(settings: Settings):
    """
    Initialize system components with proper error handling.
    
    Args:
        settings: Application settings
        
    Returns:
        Tuple[StorageManager, RetrievalQA, WebScraper]: Initialized components
        
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
            index_name=settings.pinecone_index_name
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
        
        return storage_manager, qa_system, scraper
        
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

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    error_msg = f"Unhandled error: {str(exc)}"
    logger.error(error_msg, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": error_msg}
    )

# Add CORS middleware with configuration
@app.on_event("startup")
async def startup_event():
    """
    Configure application on startup.
    
    This function:
    1. Configures logging based on settings
    2. Sets up CORS middleware
    3. Initializes system components
    
    Raises:
        RuntimeError: If application startup fails
    """
    try:
        settings = get_settings()
        
        # Configure logging level from settings
        logger.setLevel(getattr(logging, settings.log_level.upper()))
        
        # Configure CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        logger.info("Starting application initialization...")
        global storage_manager, qa_system, scraper
        storage_manager, qa_system, scraper = initialize_components(settings)
        logger.info("Application initialization completed successfully")
        
    except Exception as e:
        error_msg = f"Failed to start application: {str(e)}"
        logger.critical(error_msg, exc_info=True)
        raise RuntimeError(error_msg)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown."""
    logger.info("Shutting down application...")
    # Add cleanup code here if needed

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
@app.post("/api/scrape", response_model=ScrapingStatus)
async def start_scraping(
    target: ScrapingTarget,
    settings: Settings = Depends(get_settings)
):
    """
    Start a web scraping job for the specified target.
    
    Args:
        target: Scraping target configuration
        settings: Application settings
        
    Returns:
        ScrapingStatus: Status of the created scraping job
        
    Raises:
        HTTPException: If scraping fails
    """
    try:
        logger.info(f"Starting scraping job for URL: {target.url}")
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if target.schedule:
            logger.info(f"Scheduling scraping job {job_id} with schedule: {target.schedule}")
            scraper.schedule_scraping(
                task=scraper.scrape_html,
                interval=3600,  # Convert schedule to interval
                url=str(target.url)
            )
            next_run = datetime.now()
        else:
            logger.info(f"Starting immediate scraping job {job_id}")
            next_run = None
        
        status = ScrapingStatus(
            job_id=job_id,
            url=target.url,
            status="scheduled" if target.schedule else "started",
            last_run=datetime.now(),
            next_run=next_run
        )
        logger.info(f"Created scraping job: {status.dict()}")
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
    settings = get_settings()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        debug=settings.debug_mode
    ) 