from typing import List
import numpy as np
from .database import collection

def store_embeddings(text_chunks: List[str], embeddings: List[np.ndarray]) -> bool:
    """
    Store text chunks and their embeddings in the vector database.
    
    Args:
        text_chunks: List of text chunks to store
        embeddings: List of embedding vectors for each chunk
        
    Returns:
        bool: True if storage was successful, False otherwise
    """
    try:
        # Validate inputs
        if len(text_chunks) != len(embeddings):
            raise ValueError("Number of text chunks must match number of embeddings")
            
        if not text_chunks or not embeddings:
            raise ValueError("Empty text chunks or embeddings provided")
            
        # Store chunks with their embeddings
        collection.add(
            ids=[str(i) for i in range(len(text_chunks))],
            embeddings=[emb.tolist() for emb in embeddings],
            documents=text_chunks
        )
        
        return True
        
    except Exception as e:
        print(f"Error storing embeddings: {str(e)}")
        return False
