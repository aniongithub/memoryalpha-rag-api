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
    def clip_model(self):
        """Lazy load CLIP model for image search."""
        if self._clip_model is None:
            logger.info("Loading CLIP model for image search...")
            self._clip_model = SentenceTransformer('clip-ViT-B-32')
            logger.info("CLIP model loaded successfully")
        return self._clip_model

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

    @property
    def image_collection(self):
        """Lazy load image collection."""
        if self._image_collection is None:
            from chromadb.utils import embedding_functions

            class CLIPEmbeddingFunction(embedding_functions.EmbeddingFunction):
                def __init__(self, clip_model):
                    self.clip_model = clip_model
                def __call__(self, input):
                    embeddings = []
                    for img in input:
                        embedding = self.clip_model.encode(img)
                        embeddings.append(embedding.tolist())
                    return embeddings

            self._clip_ef = CLIPEmbeddingFunction(self.clip_model)
            self._image_collection = self.client.get_or_create_collection("memoryalpha_images", embedding_function=self._clip_ef)
        return self._image_collection

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

    def search_image(self, image_path: str, top_k: int = 5, 
                     model: str = os.getenv("DEFAULT_IMAGE_MODEL")) -> Dict[str, Any]:
        """
        Search for images similar to the provided image.
        """
        from PIL import Image
        import requests
        import tempfile
        import os

        if not model:
            raise ValueError("model must be provided or set in DEFAULT_IMAGE_MODEL environment variable.")

        try:
            # Load image and generate embedding
            image = Image.open(image_path).convert('RGB')
            image_embedding = self.clip_model.encode(image)
            image_embedding = image_embedding.tolist()

            # Search image collection
            image_results = self.image_collection.query(
                query_embeddings=[image_embedding],
                n_results=top_k
            )

            # Process results
            if not image_results["documents"] or not image_results["documents"][0]:
                return {"model_answer": "No matching visual records found in Starfleet archives."}

            # Format results for the model
            formatted_results = []
            for i, (doc, meta, dist) in enumerate(zip(
                image_results['documents'][0],
                image_results['metadatas'][0],
                image_results['distances'][0]
            ), 1):
                record_name = meta.get('image_name', 'Unknown visual record')
                formatted_results.append(f"Visual Record {i}: {record_name}")

            result_text = "\n".join(formatted_results)

            # Use LLM to provide a natural language summary
            prompt = f"""You are an LCARS computer system analyzing visual records from Starfleet archives.

Based on these visual record matches, identify what subject or scene is being depicted:

{result_text}

Provide a direct identification of the subject without referencing images, searches, or technical processes. Stay in character as an LCARS computer system."""

            result = self.ollama_client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an LCARS computer system. Respond in character without breaking the Star Trek universe immersion. Do not reference images, searches, or technical processes."},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                options={"temperature": 0.3, "num_predict": 200}
            )

            return {"model_answer": result['message']['content']}

        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return {"model_answer": "Error accessing visual records database."}