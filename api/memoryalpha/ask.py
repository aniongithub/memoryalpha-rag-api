from fastapi import APIRouter, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
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
    use_tools: Optional[bool] = False

@router.post("/memoryalpha/rag/ask")
def ask_endpoint_post(request: AskRequest):
    """
    Query the RAG pipeline and return the full response.
    Accepts POST requests with JSON payload for cleaner API usage.
    """
    try:
        result = rag_instance.ask(
            request.question, 
            max_tokens=request.max_tokens,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            use_tools=request.use_tools,
        )
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/memoryalpha/rag/ask")
def ask_endpoint(
    question: str = Query(..., description="The user question"),
    max_tokens: int = Query(2048, description="Maximum tokens to generate"),
    top_k: int = Query(10, description="Number of documents to retrieve"),
    top_p: float = Query(0.8, description="Sampling parameter"),
    temperature: float = Query(0.3, description="Randomness/creativity of output"),
    use_tools: bool = Query(False, description="Use the legacy tool-calling agent loop instead of single-pass RAG"),
):
    """
    Query the RAG pipeline and return the full response.
    Uses single-pass (no tool-calling) RAG by default; set use_tools=true for the legacy loop.
    """
    try:
        result = rag_instance.ask(
            question, 
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            use_tools=use_tools,
        )
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/memoryalpha/rag/stream")
def stream_endpoint_post(request: AskRequest):
    """Stream the answer as it is generated (single-pass RAG). Returns text/plain chunks."""
    generator = rag_instance.ask_stream(
        request.question,
        max_tokens=request.max_tokens,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
    )
    return StreamingResponse(generator, media_type="text/plain; charset=utf-8")

@router.get("/memoryalpha/rag/stream")
def stream_endpoint(
    question: str = Query(..., description="The user question"),
    max_tokens: int = Query(2048, description="Maximum tokens to generate"),
    top_k: int = Query(10, description="Number of documents to retrieve"),
    top_p: float = Query(0.8, description="Sampling parameter"),
    temperature: float = Query(0.3, description="Randomness/creativity of output"),
):
    """Stream the answer as it is generated (single-pass RAG). Returns text/plain chunks."""
    generator = rag_instance.ask_stream(
        question,
        max_tokens=max_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    return StreamingResponse(generator, media_type="text/plain; charset=utf-8")
