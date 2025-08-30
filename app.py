from fastapi import FastAPI, Header, HTTPException, Depends
from typing import Optional, List
from pydantic import BaseModel
import os

from rag_app import RAGPipeline

app = FastAPI(title="RAG API")

API_TOKEN = os.getenv("API_KEY", "my-secret-token")

def verify_token(x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

class IndexRequest(BaseModel):
    urls: List[str]

class ChatRequest(BaseModel):
    question: str
    k: Optional[int] = 5

rag = RAGPipeline()

@app.post("/api/v1/index", dependencies=[Depends(verify_token)])
async def index_data(request: IndexRequest):
    stats = rag.build_knowledge_base(request.urls)
    return {
        "status": "Knowledge base built successfully",
        "total_documents": stats["total_documents"],
        "total_chunks": stats["total_chunks"],
        "failed_urls": stats["failed_urls"],
        "embedding_dimension": stats["embedding_dimension"]
    }

@app.post("/api/v1/chat", dependencies=[Depends(verify_token)])
async def chat(request: ChatRequest):
    if rag.retriever is None:
        rag.load()
    result = rag.query(request.question, k=request.k)
    return result
