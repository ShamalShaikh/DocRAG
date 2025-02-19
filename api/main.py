import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from api.routes import retrieval, ingest
from src.database import collection
from src.generate_embeddings import model
import ollama

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF-RAG API",
    description="API for PDF document processing and querying using RAG",
    version="1.0.0",
    debug=True
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with proper path configuration
app.include_router(
    retrieval.router,
    prefix="/query",
    tags=["Query"]
)
app.include_router(
    ingest.router,
    prefix="/ingest",
    tags=["Ingestion"]
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint that tests core components."""
    try:
        logger.debug("Starting health check")
        status = {
            "api": "healthy",
            "database": False,
            "embedding_model": False,
            "llm": False
        }
        
        # Test database
        logger.debug("Testing database...")
        collection.count()
        status["database"] = True
        logger.info("Database check passed")
        
        # Test embedding model
        logger.debug("Testing embedding model...")
        model.encode(["test"])
        status["embedding_model"] = True
        logger.info("Embedding model check passed")
        
        # Test LLM
        logger.debug("Testing LLM...")
        ollama.list()
        status["llm"] = True
        logger.info("LLM check passed")
        
        logger.info("All health checks passed")
        return status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )
