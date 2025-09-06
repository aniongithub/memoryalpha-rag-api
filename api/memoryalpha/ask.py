from fastapi import APIRouter, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from .rag import MemoryAlphaRAG

router = APIRouter()

# Singleton or global instance for demo; in production, manage lifecycle properly
rag_instance = MemoryAlphaRAG()

class AskRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 2048
    top_k: Optional[int] = 10
    top_p: Optional[float] = 0.8
    temperature: Optional[float] = 0.3

@router.post("/memoryalpha/rag/ask")
def ask_endpoint_post(request: AskRequest):
    """
    Query the RAG pipeline and return the full response.
    Accepts POST requests with JSON payload for cleaner API usage.
    """
    try:
        answer = rag_instance.ask(
            request.question, 
            max_tokens=request.max_tokens,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature
        )
        return JSONResponse(content={"response": answer})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/memoryalpha/rag/ask")
def ask_endpoint(
    question: str = Query(..., description="The user question"),
    max_tokens: int = Query(2048, description="Maximum tokens to generate"),
    top_k: int = Query(10, description="Number of documents to retrieve"),
    top_p: float = Query(0.8, description="Sampling parameter"),
    temperature: float = Query(0.3, description="Randomness/creativity of output")
):
    """
    Query the RAG pipeline and return the full response.
    Now uses advanced tool-enabled RAG by default for better results.
    """
    try:
        answer = rag_instance.ask(
            question, 
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )
        return JSONResponse(content={"response": answer})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
