from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME, cache_folder="./cache")


def generate_embeddings(text_chunks):
    return model.encode(text_chunks, show_progress_bar=True)
