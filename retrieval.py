import numpy as np
from generate_embeddings import model
from store_vector_db import collection

def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()], 
        n_results=top_k
    )
    return results["documents"][0]
