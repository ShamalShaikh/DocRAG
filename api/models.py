from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3  # Optional parameter for number of chunks to retrieve

class QueryResponse(BaseModel):
    query: str
    answer: str
    source_chunks: list[str] = []  # Optional: include source chunks

class IngestResponse(BaseModel):
    filename: str
    message: str
    num_chunks: int
