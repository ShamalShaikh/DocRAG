from functools import lru_cache
from src.database import collection
from src.generate_embeddings import model

@lru_cache()
def get_collection():
    """Returns a cached instance of the ChromaDB collection."""
    return collection

@lru_cache()
def get_embedding_model():
    """Returns a cached instance of the embedding model."""
    return model
