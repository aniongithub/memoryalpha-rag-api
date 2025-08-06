from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from .rag import MemoryAlphaRAG, ThinkingMode

router = APIRouter()

# Singleton or global instance for demo; in production, manage lifecycle properly
rag_instance = MemoryAlphaRAG()
ThinkingMode = ThinkingMode

@router.get("/memoryalpha/rag/ask")
def ask_endpoint(
    question: str = Query(..., description="The user question"),
    thinkingmode: str = Query("DISABLED", description="Thinking mode: DISABLED, QUIET, or VERBOSE"),
    max_tokens: int = Query(2048, description="Maximum tokens to generate"),
    top_k: int = Query(10, description="Number of documents to retrieve"),
    top_p: float = Query(0.8, description="Sampling parameter"),
    temperature: float = Query(0.3, description="Randomness/creativity of output")
):
    """
    Query the RAG pipeline and return the full response (including thinking if enabled).
    """
    try:
        # Set the thinking mode for this request
        rag_instance.thinking_mode = ThinkingMode[thinkingmode.upper()]
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
