import os
from extract_text import extract_text_from_pdf
from chunk_text import chunk_text
from generate_embeddings import generate_embeddings
from store_vector_db import store_embeddings, check_existing_text
from database import collection

# Process PDFs
data_dir = "data"
pdf_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]

for pdf_path in pdf_paths:
    print(f"Processing {pdf_path}...")

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    print("Extracted text.")

    # Chunk the text
    chunks = chunk_text(text)
    print(f"Chunked into {len(chunks)} pieces.")

    # Remove duplicates by checking existing data in ChromaDB
    unique_chunks = [chunk for chunk in chunks if not check_existing_text(chunk, collection)]
    print(f"Filtered {len(chunks) - len(unique_chunks)} duplicate chunks.")

    if unique_chunks:
        # Generate embeddings for unique chunks
        embeddings = generate_embeddings(unique_chunks)
        store_embeddings(unique_chunks, embeddings, collection)
        print(f"Stored {len(unique_chunks)} new text chunks in ChromaDB.")
    else:
        print("No new unique data to store.")

print("Processing complete.")
