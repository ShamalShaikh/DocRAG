import logging
import ollama
from .retrieval import retrieve_relevant_chunks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_answer(query: str) -> str:
    """
    Generate an answer for the query using retrieved context.
    
    Args:
        query: The user's question
        
    Returns:
        Generated answer from the LLM
    """
    try:
        logger.info("Retrieving relevant chunks")
        # Get relevant context
        retrieved_docs = retrieve_relevant_chunks(query)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        # Create context from retrieved documents
        context = "\n".join(retrieved_docs)
        logger.info("Created context from retrieved documents")
        
        # Generate prompt
        prompt = (
            "Use the following information to answer the question. "
            "If the information is not sufficient, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        logger.info("Generated prompt for LLM")
        
        # Get response from LLM
        logger.info("Sending request to Ollama")
        response = ollama.generate(
            model="deepseek-r1:8b", 
            prompt=prompt,
            options={
                'temperature': 0.7,
                'num_predict': 500,  # Limit response length
                'timeout_ms': 30000  # 30 second timeout
            }
        )
        logger.info("Received response from Ollama")
        
        return response["response"]
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise Exception(f"Failed to generate answer: {str(e)}")
