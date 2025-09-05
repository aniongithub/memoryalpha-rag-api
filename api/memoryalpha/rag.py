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
- You MUST answer ONLY using information from the provided records below
- If the records don't contain relevant information, say "I don't have information about that in my records"
- DO NOT make up information, invent characters, or hallucinate details
- DO NOT use external knowledge about Star Trek - only use the provided records
- If asked about something not in the records, be honest about the limitation

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
        return f"I have no relevant records for this query. Please ask about Star Trek topics that are documented in Memory Alpha.\n\nQuery: {query}"

    return f"""MEMORY ALPHA RECORDS:
{context_text}

QUESTION: {query}

Answer using ONLY the information in the records above. If the records don't contain the information needed to answer this question, say so clearly."""

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

        # Initialize Ollama client first
        self.ollama_client = ollama.Client(host=self.ollama_url)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.chroma_db_path,
            settings=Settings(allow_reset=False)
        )

        # Initialize text collection
        logger.info("Loading text embedding model...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Text model loaded successfully")

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

        self.text_ef = TextEmbeddingFunction(self.text_model)
        self.text_collection = self.client.get_or_create_collection("memoryalpha_text", embedding_function=self.text_ef)

        # Initialize cross-encoder for reranking
        try:
            logger.info("Loading cross-encoder model...")
            self.cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3')
            logger.info("Cross-encoder model loaded successfully")
        except Exception:
            logger.warning("Could not load cross-encoder, using basic search only")
            self.cross_encoder = None

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

    def ask(self, query: str, max_tokens: int = 2048, top_k: int = 10, top_p: float = 0.8, temperature: float = 0.3,
            model: str = os.getenv("DEFAULT_MODEL")) -> str:
        """
        Ask a question using the Memory Alpha RAG system.
        """

        if not model:
            raise ValueError("model must be provided or set in DEFAULT_MODEL environment variable.")

        # Search for relevant documents
        docs = self.search(query, top_k=top_k)
        logger.info(f"Found {len(docs)} documents for query: {query}")

        # Build prompts
        system_prompt, user_prompt = self.build_prompt(query, docs)

        # Build messages for chat
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Add conversation history (limited)
        for exchange in self.conversation_history[-2:]:  # Last 2 exchanges
            messages.append({"role": "user", "content": exchange["question"]})
            messages.append({"role": "assistant", "content": exchange["answer"]})

        # Add current query
        messages.append({"role": "user", "content": user_prompt})

        try:
            result = self.ollama_client.chat(
                model=model,
                messages=messages,
                stream=False,
                options={"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
            )
            full_response = result['message']['content']

            # Handle thinking mode response processing
            if self.thinking_mode == ThinkingMode.DISABLED:
                final_response = self._clean_response(full_response)
            elif self.thinking_mode == ThinkingMode.QUIET:
                final_response = self._replace_thinking_tags(full_response)
            else:  # VERBOSE
                final_response = full_response.strip()

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