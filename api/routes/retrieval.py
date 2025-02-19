import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Depends, HTTPException
from api.models import QueryRequest, QueryResponse
from api.dependencies import get_collection, get_embedding_model
from src.augment_llm import generate_answer
from src.retrieval import retrieve_relevant_chunks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=3)

# Constants
TIMEOUT_SECONDS = 60

router = APIRouter()

async def run_in_threadpool(func, *args, **kwargs):
    try:
        return await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                thread_pool, 
                lambda: func(*args, **kwargs)
            ),
            timeout=TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {TIMEOUT_SECONDS} seconds")
        raise HTTPException(
            status_code=504,
            detail=f"Operation timed out after {TIMEOUT_SECONDS} seconds"
        )

async def async_retrieve_and_generate(query: str, top_k: int) -> tuple[list[str], str]:
    """Async wrapper for retrieval and generation"""
    try:
        # Run CPU-intensive tasks in thread pool
        chunks = await run_in_threadpool(
            retrieve_relevant_chunks,
            query,
            top_k
        )
        
        answer = await run_in_threadpool(
            generate_answer,
            query
        )
        
        return chunks, answer
    except Exception as e:
        logger.error(f"Error in async_retrieve_and_generate: {str(e)}")
        raise

@router.post("", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    collection=Depends(get_collection),
    model=Depends(get_embedding_model)
):
    """Query the RAG system with a natural language question."""
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Set timeout for the entire operation
        async with asyncio.timeout(30):  # 30 second timeout
            chunks, answer = await async_retrieve_and_generate(
                request.query, request.top_k
            )
        
        logger.info("Successfully generated response")
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            source_chunks=chunks
        )
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        raise HTTPException(
            status_code=504,
            detail="Request timed out after 30 seconds"
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
