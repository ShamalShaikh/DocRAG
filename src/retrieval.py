import logging
import numpy as np
from .generate_embeddings import model
from .store_vector_db import collection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrieve_relevant_chunks(query: str, top_k: int = 3) -> list[str]:
    """
    Retrieve the most relevant text chunks for a given query.
    
    Args:
        query: The search query
        top_k: Number of chunks to retrieve
        
    Returns:
        List of relevant text chunks
    """
    try:
        logger.info(f"Generating embedding for query: {query}")
        # Generate query embedding
        query_embedding = model.encode([query])[0]
        logger.info("Successfully generated query embedding")
        
        # Query the collection
        logger.info(f"Querying collection for top {top_k} results")
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        logger.info(f"Retrieved {len(results['documents'][0])} chunks")
        
        return results["documents"][0]
    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        raise Exception(f"Failed to retrieve chunks: {str(e)}")
