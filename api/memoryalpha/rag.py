from __future__ import annotations

import os
import sys
import re
import logging
import warnings
from typing import List, Dict, Any

# External modules
from sentence_transformers import CrossEncoder, SentenceTransformer
import ollama

# RAG components
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*is deprecated.*", category=FutureWarning)

def get_user_prompt(context_text: str, query: str) -> str:
    """Format user prompt with context and query"""

    if not context_text.strip():
        return f"Starfleet database records contain no relevant information for this inquiry. Please inquire about documented Star Trek topics.\n\nINQUIRY: {query}"

    return f"""MEMORY ALPHA RECORDS:
{context_text}

INQUIRY: {query}

Accessing Starfleet database records. Provide analysis using ONLY the information in the records above. If the records don't contain the information needed to answer this inquiry, state that the information is not available in current records."""

class MemoryAlphaRAG:
    def __init__(self,
                 chroma_db_path: str = os.getenv("DB_PATH"),
                 ollama_url: str = os.getenv("OLLAMA_URL"),
                 collection_name: str = os.getenv("COLLECTION_NAME", "memoryalpha"),
                 max_history_turns: int = 5):

        if not chroma_db_path:
            raise ValueError("chroma_db_path must be provided or set in CHROMA_DB_PATH environment variable.")
        if not ollama_url:
            raise ValueError("ollama_url must be provided or set in OLLAMA_URL environment variable.")

        self.chroma_db_path = chroma_db_path
        self.ollama_url = ollama_url
        self.collection_name = collection_name
        self.max_history_turns = max_history_turns
        self.conversation_history: List[Dict[str, str]] = []

        # Initialize lightweight components
        self.ollama_client = ollama.Client(host=self.ollama_url)
        self.client = chromadb.PersistentClient(
            path=self.chroma_db_path,
            settings=Settings(allow_reset=False)
        )

        # Lazy-loaded components
        self._text_model = None
        self._cross_encoder = None
        self._clip_model = None
        self._text_collection = None
        self._image_collection = None
        self._text_ef = None
        self._clip_ef = None

    @property
    def text_model(self):
        """Lazy load text embedding model."""
        if self._text_model is None:
            logger.info("Loading text embedding model...")
            self._text_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Text model loaded successfully")
        return self._text_model

    @property
    def cross_encoder(self):
        """Lazy load cross-encoder model."""
        if self._cross_encoder is None:
            try:
                logger.info("Loading cross-encoder model...")
                self._cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3')
                logger.info("Cross-encoder model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load cross-encoder: {e}")
                self._cross_encoder = None
        return self._cross_encoder

    @property
    def text_collection(self):
        """Lazy load text collection."""
        if self._text_collection is None:
            from chromadb.utils import embedding_functions

            class TextEmbeddingFunction(embedding_functions.EmbeddingFunction):
                def __init__(self, text_model):
                    self.text_model = text_model
                def __call__(self, input):
                    embeddings = []
                    for text in input:
                        embedding = self.text_model.encode(text)
                        embeddings.append(embedding.tolist())
                    return embeddings

            self._text_ef = TextEmbeddingFunction(self.text_model)
            self._text_collection = self.client.get_or_create_collection("memoryalpha_text", embedding_function=self._text_ef)
        return self._text_collection

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search the Memory Alpha database for relevant documents."""

        try:
            # Perform semantic search
            results = self.text_collection.query(
                query_texts=[query],
                n_results=min(top_k * 2, 50)  # Get more results for reranking
            )

            if not results["documents"] or not results["documents"][0]:
                logger.warning(f"No documents found for query: {query}")
                return []

            docs = []
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                docs.append({
                    "content": doc,
                    "title": meta.get("title", "Unknown"),
                    "distance": dist
                })

            # Re-rank using cross-encoder if available
            if self.cross_encoder and len(docs) > top_k:
                logger.info("Re-ranking results with cross-encoder")
                # Limit to top candidates for re-ranking to avoid performance issues
                rerank_candidates = docs[:min(len(docs), top_k + 5)]  # Only re-rank top candidates
                
                # Prepare pairs for cross-encoder with truncated content
                pairs = []
                for doc in rerank_candidates:
                    content = doc['content']
                    if len(content) > 512:  # Truncate long content for cross-encoder
                        content = content[:512]
                    pairs.append([query, content])
                
                try:
                    scores = self.cross_encoder.predict(pairs)
                    
                    # Sort by cross-encoder scores (higher is better)
                    ranked_docs = sorted(zip(rerank_candidates, scores), key=lambda x: x[1], reverse=True)
                    reranked = [doc for doc, score in ranked_docs]
                    
                    # Replace original docs with re-ranked ones
                    docs = reranked + docs[len(rerank_candidates):]
                    logger.info(f"Cross-encoder re-ranking completed, top score: {scores[0]:.4f}")
                except Exception as e:
                    logger.warning(f"Cross-encoder re-ranking failed: {e}, using original ranking")
                    # Continue with original docs if re-ranking fails
            return docs[:top_k]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def build_prompt(self, query: str, docs: List[Dict[str, Any]]) -> tuple[str, str]:
        """Build the prompt with retrieved documents."""

        system_prompt = """You are an LCARS computer system with access to Star Trek Memory Alpha records.

CRITICAL INSTRUCTIONS:
- You MUST answer ONLY using information from the provided records
- If the records don't contain relevant information, say "I don't have information about that in my records"
- DO NOT make up information, invent characters, or hallucinate details
- DO NOT use external knowledge about Star Trek - only use the provided records
- AVOID mirror universe references unless specifically asked about it
- If asked about something not in the records, be honest about the limitation
- Stay in character as an LCARS computer system at all times

Answer directly in a single paragraph."""

        if not docs:
            context_text = ""
        else:
            # Format context with clear structure
            context_parts = []
            for i, doc in enumerate(docs, 1):
                content = doc['content']
                # Limit content length to avoid token limits
                if len(content) > 1000:
                    content = content[:1000] + "..."
                context_parts.append(f"DOCUMENT {i}: {doc['title']}\n{content}")

            context_text = "\n\n".join(context_parts)

        user_prompt = get_user_prompt(context_text, query)
        return system_prompt, user_prompt

    def search_tool(self, query: str, top_k: int = 5) -> str:
        """
        Tool function that the LLM can call to search the database.
        Returns formatted search results as a string.
        """
        logger.info(f"Search tool called with query: '{query}', top_k: {top_k}")
        docs = self.search(query, top_k=top_k)
        logger.info(f"Search returned {len(docs)} documents")
        
        if not docs:
            logger.warning(f"No documents found for query: {query}")
            return f"No relevant documents found for query: {query}"
        
        results = []
        for i, doc in enumerate(docs, 1):
            content = doc['content']
            if len(content) > 500:  # Limit content for tool responses
                content = content[:500] + "..."
            results.append(f"DOCUMENT {i}: {doc['title']}\n{content}")
        
        formatted_result = f"Search results for '{query}':\n\n" + "\n\n".join(results)
        logger.info(f"Formatted search result length: {len(formatted_result)}")
        return formatted_result

    def ask(self, query: str, max_tokens: int = 2048, top_k: int = 10, top_p: float = 0.8, temperature: float = 0.3,
            model: str = os.getenv("DEFAULT_MODEL")) -> str:
        """
        Ask a question using the advanced Memory Alpha RAG system with tool use.
        """

        if not model:
            raise ValueError("model must be provided or set in DEFAULT_MODEL environment variable.")

        logger.info(f"Starting tool-enabled RAG for query: {query}")

        # Define the search tool
        search_tool_definition = {
            "type": "function",
            "function": {
                "name": "search_memory_alpha",
                "description": "Search the Star Trek Memory Alpha database for information. Use this tool when you need to find specific information about Star Trek characters, episodes, ships, planets, or other topics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant information"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of documents to retrieve (default: 5, max: 10)",
                            "default": 5,
                            "maximum": 10
                        }
                    },
                    "required": ["query"]
                }
            }
        }

        system_prompt = """You are an LCARS computer system with access to Star Trek Memory Alpha records.

You have access to a search tool that can query the Memory Alpha database. You MUST use this tool for ALL questions about Star Trek.

CRITICAL REQUIREMENTS:
- You MUST call the search tool for EVERY question
- You cannot answer any question without first using the search tool
- Do NOT use any external knowledge or make up information
- Only answer based on the search results provided
- If no relevant information is found, say so clearly
- ALWAYS provide a final answer after using tools - do not just think without concluding

TOOL USAGE:
- Always call the search tool first, before attempting to answer
- Do NOT directly use the input question, only use keywords from it
- Use only key terms from the input question for seaching
- If insufficient information is found on the first try, retry with variations or relevant info from previous queries
- DISCARD details from alternate universes, books or timelines
- DISREGARD details about books, comics, or non-canon sources
- NEVER mention appearances or actors, only in-universe details
- Ensure a complete answer can be formulated before stopping searches
- Wait for search results before providing your final answer

RESPONSE FORMAT:
- Use tools when needed
- Provide your final answer clearly and concisely
- Do not add details that are irrelevant to the question
- Stay in-character as an LCARS computer system at all times, do not allude to the Star Trek universe itself or it being a fictional setting
- Do not mention the search results, only the final in-universe answer"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please answer this question about Star Trek: {query}"}
        ]

        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        has_used_tool = False

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration} for query: {query}")
            
            try:
                logger.info(f"Sending messages to LLM: {[msg['role'] for msg in messages]}")
                result = self.ollama_client.chat(
                    model=model,
                    messages=messages,
                    stream=False,
                    think=False,
                    options={"temperature": temperature, "top_p": top_p, "num_predict": max_tokens},
                    tools=[search_tool_definition]
                )
                
                response_message = result['message']
                logger.info(f"LLM response type: {type(response_message)}")
                logger.debug(f"Response message content: {response_message.get('content', 'No content')[:200]}...")
                
                # Check if the model wants to use a tool
                tool_calls = getattr(response_message, 'tool_calls', None) or response_message.get('tool_calls')
                if tool_calls:
                    has_used_tool = True
                    logger.info(f"Tool calls detected: {len(tool_calls)}")
                    # Execute the tool call
                    tool_call = tool_calls[0]
                    logger.info(f"Tool call: {tool_call.get('function', {}).get('name', 'Unknown')}")
                    
                    if tool_call.get('function', {}).get('name') == 'search_memory_alpha':
                        args = tool_call.get('function', {}).get('arguments', {})
                        search_query = args.get('query', '')
                        search_top_k = min(args.get('top_k', 5), 10)
                        
                        logger.info(f"Executing search for: '{search_query}' with top_k={search_top_k}")
                        
                        # Execute the search
                        search_result = self.search_tool(search_query, search_top_k)
                        logger.info(f"Search result length: {len(search_result)}")
                        logger.debug(f"Search result preview: {search_result[:500]}...")
                        
                        # Add the tool call and result to messages
                        messages.append(response_message)
                        messages.append({
                            "role": "tool",
                            "content": search_result,
                            "tool_call_id": tool_call.get('id', '')
                        })
                        
                        logger.info("Continuing conversation with tool results")
                        continue  # Continue the conversation with tool results
                
                # If no tool call and we haven't used tools yet, force a search
                if not has_used_tool and iteration == 1:
                    logger.info("LLM didn't use tool on first attempt, forcing initial search")
                    search_result = self.search_tool(query, 5)
                    messages.append({
                        "role": "tool",
                        "content": search_result,
                        "tool_call_id": "forced_search"
                    })
                    has_used_tool = True
                    continue
                
                # If no tool call, this is the final answer
                final_response = response_message.get('content', '')
                if not final_response:
                    logger.warning("LLM returned empty content")
                    final_response = "I apologize, but I was unable to generate a response."
                    
                logger.info(f"Final response length: {len(final_response)}")
                logger.info(f"Final response preview: {final_response[:200]}...")
                logger.debug(f"Raw final response: {repr(final_response[:500])}")
                
                # Remove ANSI codes and LCARS prefix
                final_response = re.sub(r"\033\[[0-9;]*m", "", final_response)
                final_response = final_response.replace("LCARS: ", "").strip()
                
                self._update_history(query, final_response)
                logger.info("Returning final answer")
                return final_response
                
            except Exception as e:
                logger.error(f"Chat failed: {e}")
                return f"Error processing query: {str(e)}"

        # Fallback if max iterations reached
        logger.warning(f"Max iterations reached for query: {query}")
        return "Query processing exceeded maximum iterations. Please try a simpler question."

    def _update_history(self, question: str, answer: str):
        """Update conversation history."""
        self.conversation_history.append({"question": question, "answer": answer})
        self.conversation_history = self.conversation_history[-self.max_history_turns:]