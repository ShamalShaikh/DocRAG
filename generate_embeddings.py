from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(MODEL_NAME, cache_folder="./cache")


def generate_embeddings(text_chunks):
    return model.encode(text_chunks, show_progress_bar=True)
