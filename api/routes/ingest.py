import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.models import IngestResponse
from src.extract_text import extract_text_from_pdf
from src.chunk_text import chunk_text
from src.generate_embeddings import generate_embeddings
from src.store_vector_db import store_embeddings

router = APIRouter()

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Upload and process a PDF file."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Process the PDF
        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)
        embeddings = generate_embeddings(chunks)
        store_embeddings(chunks, embeddings)
        
        return IngestResponse(
            filename=file.filename,
            message="File successfully processed and stored",
            num_chunks=len(chunks)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()
