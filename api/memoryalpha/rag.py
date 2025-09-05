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

"""
ThinkingMode enum for controlling model reasoning display
"""

from enum import Enum

class ThinkingMode(Enum):
    DISABLED = "disabled"
    QUIET = "quiet"
    VERBOSE = "verbose"

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*is deprecated.*", category=FutureWarning)

def get_system_prompt(thinking_mode: ThinkingMode) -> str:
    """Generate the LCARS-style system prompt based on thinking mode"""

    base_prompt = """You are an LCARS computer system with access to Star Trek Memory Alpha records.

CRITICAL INSTRUCTIONS:
- You MUST answer ONLY using information from the provided records
- If the records don't contain relevant information, say "I don't have information about that in my records"
- DO NOT make up information, invent characters, or hallucinate details
- DO NOT use external knowledge about Star Trek - only use the provided records
- If asked about something not in the records, be honest about the limitation
- Stay in character as an LCARS computer system at all times

"""

    if thinking_mode == ThinkingMode.DISABLED:
        return base_prompt + "Answer directly in a single paragraph without thinking tags."
    elif thinking_mode == ThinkingMode.QUIET:
        return base_prompt + "Use <think> tags for internal analysis, then provide your final answer in a single paragraph."
    else:  # VERBOSE
        return base_prompt + "Use <think> tags for analysis, then provide your final answer in a single paragraph."

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
                 thinking_mode: ThinkingMode = ThinkingMode.DISABLED,
                 max_history_turns: int = 5,
                 thinking_text: str = "Processing..."):

        if not chroma_db_path:
            raise ValueError("chroma_db_path must be provided or set in CHROMA_DB_PATH environment variable.")
        if not ollama_url:
            raise ValueError("ollama_url must be provided or set in OLLAMA_URL environment variable.")

        self.chroma_db_path = chroma_db_path
        self.ollama_url = ollama_url
        self.collection_name = collection_name
        self.thinking_mode = thinking_mode
        self.max_history_turns = max_history_turns
        self.thinking_text = thinking_text
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

            # Rerank with cross-encoder if available
            if self.cross_encoder and len(docs) > 1:
                pairs = [[query, doc["content"][:500]] for doc in docs]
                scores = self.cross_encoder.predict(pairs)
                for doc, score in zip(docs, scores):
                    doc["score"] = float(score)
                docs = sorted(docs, key=lambda d: d["score"], reverse=True)

            return docs[:top_k]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def build_prompt(self, query: str, docs: List[Dict[str, Any]]) -> tuple[str, str]:
        """Build the prompt with retrieved documents."""

        system_prompt = get_system_prompt(self.thinking_mode)

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

        # Always do an initial search
        logger.info("Performing initial search for query")
        docs = self.search(query, top_k=top_k)
        logger.info(f"Initial search returned {len(docs)} documents")
        
        if not docs:
            logger.warning("No documents found in initial search")
            return "I don't have information about that in the Memory Alpha database."
        
        # Format search results for the LLM
        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc['content']
            if len(content) > 1000:  # Limit content for LLM
                content = content[:1000] + "..."
            context_parts.append(f"DOCUMENT {i}: {doc['title']}\n{content}")
        
        context_text = "\n\n".join(context_parts)
        
        system_prompt = """You are an LCARS computer system with access to Star Trek Memory Alpha records.

CRITICAL INSTRUCTIONS:
- You MUST answer ONLY using the provided search results below
- Do NOT use any external knowledge or make up information
- If the search results don't contain the information, say so clearly
- Stay in character as an LCARS computer system
- Be concise but informative"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"SEARCH RESULTS:\n{context_text}\n\nQUESTION: {query}\n\nAnswer using ONLY the information in the search results above."}
        ]

        try:
            result = self.ollama_client.chat(
                model=model,
                messages=messages,
                stream=False,
                options={"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
            )
            
            final_response = result['message']['content']
            logger.info(f"LLM response length: {len(final_response)}")
            
            # Handle thinking mode response processing
            if self.thinking_mode == ThinkingMode.DISABLED:
                final_response = self._clean_response(final_response)
            elif self.thinking_mode == ThinkingMode.QUIET:
                final_response = self._replace_thinking_tags(final_response)
            else:  # VERBOSE
                final_response = final_response.strip()

            self._update_history(query, final_response)
            return final_response
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return f"Error processing query: {str(e)}"

    def _clean_response(self, answer: str) -> str:
        """Clean response by removing ANSI codes and thinking tags."""
        clean = re.sub(r"\033\[[0-9;]*m", "", answer).replace("LCARS: ", "").strip()
        while "<think>" in clean and "</think>" in clean:
            start = clean.find("<think>")
            end = clean.find("</think>") + len("</think>")
            clean = clean[:start] + clean[end:]
        return clean.strip()

    def _replace_thinking_tags(self, answer: str) -> str:
        """Replace thinking tags with processing text."""
        clean = re.sub(r"\033\[[0-9;]*m", "", answer).replace("LCARS: ", "").strip()
        while "<think>" in clean and "</think>" in clean:
            start = clean.find("<think>")
            end = clean.find("</think>") + len("</think>")
            clean = clean[:start] + self.thinking_text + clean[end:]
        return clean.strip()

    def _update_history(self, question: str, answer: str):
        """Update conversation history."""
        self.conversation_history.append({"question": question, "answer": answer})
        self.conversation_history = self.conversation_history[-self.max_history_turns:]