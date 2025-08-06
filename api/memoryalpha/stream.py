from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import json
import re

from .rag import MemoryAlphaRAG, ThinkingMode

router = APIRouter()

# Singleton or global instance for demo; in production, manage lifecycle properly
rag_instance = MemoryAlphaRAG()

@router.get("/memoryalpha/rag/stream")
def stream_endpoint(
    question: str = Query(..., description="The user question"),
    thinkingmode: str = Query("DISABLED", description="Thinking mode: DISABLED, QUIET, or VERBOSE"),
    max_tokens: int = Query(2048, description="Maximum tokens to generate"),
    top_k: int = Query(10, description="Number of documents to retrieve"),
    top_p: float = Query(0.8, description="Sampling parameter"),
    temperature: float = Query(0.3, description="Randomness/creativity of output")
):
    """
    Query the RAG pipeline and return streaming response chunks.
    """
    
    def generate_stream():
        try:
            # Set the thinking mode for this request
            rag_instance.thinking_mode = ThinkingMode[thinkingmode.upper()]
            
            # Get documents for context
            docs = rag_instance.search(question, top_k=top_k)
            system_prompt, user_prompt = rag_instance.build_prompt(question, docs)
            
            # Build messages for chat
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history
            for exchange in rag_instance.conversation_history[-3:]:  # Last 3 exchanges
                messages.append({"role": "user", "content": exchange["question"]})
                messages.append({"role": "assistant", "content": exchange["answer"]})
            
            # Add current query
            messages.append({"role": "user", "content": user_prompt})

            full_response = ""
            
            # Stream the response
            for chunk in rag_instance.ollama_client.chat(
                model=rag_instance.model,
                messages=messages,
                stream=True,
                options={"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    full_response += content
                    
                    # Send chunk as JSON
                    chunk_data = {"chunk": content}
                    yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Process final response based on thinking mode
            if rag_instance.thinking_mode == ThinkingMode.DISABLED:
                final_response = rag_instance._clean_response(full_response)
            elif rag_instance.thinking_mode == ThinkingMode.QUIET:
                final_response = rag_instance._replace_thinking_tags(full_response)
            else:  # VERBOSE
                final_response = full_response.strip()
            
            # Update history with final processed response
            rag_instance._update_history(question, final_response)
            
            # Send completion signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            # Send error
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
