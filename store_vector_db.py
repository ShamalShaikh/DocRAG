from database import collection


def store_embeddings(text_chunks, embeddings):
    for i, (text, embedding) in enumerate(zip(text_chunks, embeddings)):
        collection.add(
            ids=[str(i)],  # Unique ID for each chunk
            embeddings=[embedding.tolist()], 
            documents=[text]
        )
