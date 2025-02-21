# PDF-RAG: Intelligent Document Analysis System

A production-ready Retrieval-Augmented Generation (RAG) system that transforms PDF documents into queryable knowledge bases using state-of-the-art language models and vector search.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Advanced Configuration](#advanced-configuration)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)

## Overview

PDF-RAG bridges the gap between static PDF documents and interactive knowledge retrieval. By combining advanced text processing, semantic search, and language models, it enables natural language querying of PDF content with high accuracy and context awareness.

Key benefits:
- Extract and process text from any PDF document
- Intelligent text chunking for optimal context preservation
- Semantic search powered by state-of-the-art embeddings
- Natural language question answering
- Efficient storage and retrieval with vector database

## Key Features

- Robust PDF text extraction using PyMuPDF
- Smart text chunking with context preservation
- High-quality embeddings using all-MiniLM-L6-v2
- Persistent vector storage with ChromaDB
- Natural language queries using Deepseek
- Automatic duplicate content detection
- Efficient batch processing
- Cached embeddings for performance

## Prerequisites

- Python 3.11 or higher
- 4GB RAM minimum (8GB recommended)
- Operating System: Windows, macOS, or Linux
- Storage: 500MB minimum for database and cache

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd pdf-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python run_test.py
```

## Usage Guide

### Processing Documents

1. Place PDF files in the `data` directory
2. Run the processing script:
```bash
python process_data.py
```

### Querying Documents

Basic query:
```python
from augment_llm import generate_answer

answer = generate_answer("What are the main conclusions?")
print(answer)
```

Advanced usage:
```python
from retrieval import retrieve_relevant_chunks

# Get relevant context chunks
chunks = retrieve_relevant_chunks("technical specifications", top_k=5)

# Process multiple documents
for pdf in pdf_files:
    process_document(pdf)
```

## Project Structure

```
pdf-rag/
├── data/               # PDF document storage
├── chroma_db/         # Vector database storage
├── cache/             # Model and embedding cache
├── process_data.py    # Main processing script
├── extract_text.py    # PDF text extraction
├── chunk_text.py      # Text chunking logic
├── generate_embeddings.py  # Embedding generation
├── store_vector_db.py # Database operations
├── database.py        # Database configuration
├── retrieval.py       # Document retrieval
├── augment_llm.py     # LLM integration
└── run_test.py        # Testing script
```

## Advanced Configuration

### Chunking Configuration
```python
# chunk_text.py
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
```

### Embedding Model Settings
```python
# generate_embeddings.py
MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = "./cache"
```

### Database Settings
```python
# database.py
DB_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_rag"
```

## Performance Optimization

- Embeddings are cached for faster subsequent runs
- Batch processing for efficient document handling
- Automatic duplicate detection reduces storage overhead
- Configurable chunk sizes for memory optimization
- Persistent database for quick restarts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Last updated: February 2025
