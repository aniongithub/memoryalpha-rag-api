from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import json
import re
import time
import logging

from .rag import MemoryAlphaRAG, ThinkingMode

router = APIRouter()
logger = logging.getLogger(__name__)

# Lazy-loaded singleton
rag_instance = None

def get_rag_instance():
    global rag_instance
    if rag_instance is None:
        logger.info("Initializing MemoryAlpha RAG instance...")
        rag_instance = MemoryAlphaRAG()
        logger.info("MemoryAlpha RAG instance initialized successfully")
    return rag_instance

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
            start_time = time.time()
            
            # Get RAG instance (lazy-loaded)
            rag = get_rag_instance()
            
            # Set the thinking mode for this request
            rag.thinking_mode = ThinkingMode[thinkingmode.upper()]
            
            # Phase 1: Document retrieval
            search_start = time.time()
            docs = rag.search(question, top_k=top_k)
            search_duration = time.time() - search_start
            
            # Phase 2: Prompt building
            prompt_start = time.time()
            system_prompt, user_prompt = rag.build_prompt(question, docs)
            
            # Build messages for chat
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history
            for exchange in rag.conversation_history[-3:]:  # Last 3 exchanges
                messages.append({"role": "user", "content": exchange["question"]})
                messages.append({"role": "assistant", "content": exchange["answer"]})
            
            # Add current query
            messages.append({"role": "user", "content": user_prompt})

            # Estimate input tokens (rough approximation)
            full_prompt = system_prompt + "\n\n" + user_prompt
            for msg in messages[1:]:  # Skip system message already included
                full_prompt += "\n" + msg["content"]
            input_tokens = len(full_prompt.split()) * 1.3  # Rough token estimate
            prompt_duration = time.time() - prompt_start

            full_response = ""
            current_buffer = ""
            in_thinking_block = False
            
            # Phase 3: LLM generation
            generation_start = time.time()
            first_token_time = None
            
            # Stream the response
            for chunk in rag.ollama_client.chat(
                model=rag.model,
                messages=messages,
                stream=True,
                options={"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    full_response += content
                    current_buffer += content
                    
                    # Track time to first token
                    if first_token_time is None and content:
                        first_token_time = time.time() - generation_start
                    
                    # Process content based on thinking mode
                    if rag.thinking_mode == ThinkingMode.DISABLED:
                        # Filter out thinking blocks in real-time
                        output_content = ""
                        i = 0
                        while i < len(current_buffer):
                            if current_buffer[i:].startswith("<think>") and not in_thinking_block:
                                in_thinking_block = True
                                i += 7  # Skip "<think>"
                            elif current_buffer[i:].startswith("</think>") and in_thinking_block:
                                in_thinking_block = False
                                i += 8  # Skip "</think>"
                                current_buffer = current_buffer[i:]
                                i = 0
                            elif not in_thinking_block:
                                output_content += current_buffer[i]
                                i += 1
                            else:
                                i += 1
                        
                        # Only update buffer with remaining unprocessed content
                        if not in_thinking_block:
                            current_buffer = ""
                        
                        # Send filtered content
                        if output_content:
                            chunk_data = {"chunk": output_content}
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                    else:
                        # For other modes, send content as-is
                        chunk_data = {"chunk": content}
                        yield f"data: {json.dumps(chunk_data)}\n\n"
            
            generation_duration = time.time() - generation_start
            
            # Phase 4: Post-processing
            processing_start = time.time()
            output_tokens = len(full_response.split()) * 1.3  # Rough token estimate
            
            # Process final response based on thinking mode
            if rag.thinking_mode == ThinkingMode.DISABLED:
                final_response = rag._clean_response(full_response)
            elif rag.thinking_mode == ThinkingMode.QUIET:
                final_response = rag._replace_thinking_tags(full_response)
            else:  # VERBOSE
                final_response = full_response.strip()
            
            # Update history with final processed response
            rag._update_history(question, final_response)
            processing_duration = time.time() - processing_start
            
            # Calculate total duration
            total_duration = time.time() - start_time
            
            # Send completion signal with comprehensive metrics
            metrics = {
                "done": True,
                "metrics": {
                    "duration_seconds": round(total_duration, 3),
                    "phase_timings": {
                        "search_seconds": round(search_duration, 3),
                        "prompt_building_seconds": round(prompt_duration, 3),
                        "generation_seconds": round(generation_duration, 3),
                        "post_processing_seconds": round(processing_duration, 3),
                        "time_to_first_token_seconds": round(first_token_time, 3) if first_token_time else None
                    },
                    "input_tokens_estimated": int(input_tokens),
                    "output_tokens_estimated": int(output_tokens),
                    "total_tokens_estimated": int(input_tokens + output_tokens),
                    "documents_retrieved": len(docs),
                    "model": rag.model
                }
            }
            yield f"data: {json.dumps(metrics)}\n\n"
            
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
