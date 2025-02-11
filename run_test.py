import os
from extract_text import extract_text_from_pdf
from chunk_text import chunk_text
from generate_embeddings import generate_embeddings
from store_vector_db import store_embeddings
from augment_llm import generate_answer

# Process and store the PDF data
data_dir = "data"
pdf_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]

# Process each PDF
for pdf_path in pdf_paths:
    print(f"Processing {pdf_path}...")

    # Extract text from PDF
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    # Chunk the text
    print("Chunking text...")
    chunks = chunk_text(text)

# Generate embeddings and store in database
print("Generating embeddings and storing in database...")
embeddings = generate_embeddings(chunks)
store_embeddings(chunks, embeddings)
print(f"Successfully processed and stored {len(chunks)} text chunks")

# Test the RAG system with a query
print("\nTesting the RAG system...")
query = "Summarize the text in a few sentences."
# "What are the key findings in the research paper?"
print(f"\nQuery: {query}")
answer = generate_answer(query)
print(f"Answer: {answer}")