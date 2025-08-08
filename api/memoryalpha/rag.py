from __future__ import annotations

import os
import sys
import json
import re
import requests
import logging
import warnings
import numpy as np
from typing import List, Dict, Any

# External modules
from sentence_transformers import CrossEncoder, SentenceTransformer
import ollama

# Optional prompt UI
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter

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
    
    if thinking_mode == ThinkingMode.DISABLED:
        return "You are an LCARS computer. Use the provided records to answer questions precisely in a single paragraph. Do not use thinking tags or analysis blocks."
    elif thinking_mode == ThinkingMode.QUIET:
        return "You are an LCARS computer. Use <think> tags for your analysis, then provide a precise answer in a single paragraph. Users will only see your final answer, not your thinking."
    else:  # VERBOSE
        return "You are an LCARS computer. Use <think> tags for your analysis, then provide a precise answer in a single paragraph. Your thinking process will be visible to users."

def get_user_prompt(context_text: str, query: str) -> str:
    """Format user prompt with context and query"""
    
    return f"""Records:
{context_text}

Query: {query}"""

class MemoryAlphaRAG:
    def __init__(self,
                 chroma_db_path: str = os.getenv("DB_PATH"),
                 ollama_url: str = os.getenv("OLLAMA_URL"),
                 collection_name: str = os.getenv("COLLECTION_NAME", "memoryalpha"),
                 rerank_method: str = "cross-encoder",
                 thinking_mode: ThinkingMode = ThinkingMode.DISABLED,
                 enable_streaming: bool = True,
                 max_history_turns: int = 5,
                 thinking_text: str = "Processing..."):

        if not chroma_db_path:
            raise ValueError("chroma_db_path must be provided or set in CHROMA_DB_PATH environment variable.")
        if not ollama_url:
            raise ValueError("ollama_url must be provided or set in OLLAMA_URL environment variable.")
        if not collection_name:
            raise ValueError("collection_name must be provided or set in COLLECTION_NAME environment variable.")

        self.chroma_db_path = chroma_db_path
        self.ollama_url = ollama_url
        self.collection_name = collection_name
        self.thinking_mode = thinking_mode
        self.enable_streaming = enable_streaming
        self.max_history_turns = max_history_turns
        self.rerank_method = rerank_method
        self.thinking_text = thinking_text
        self.conversation_history: List[Dict[str, str]] = []

        # Initialize conversation messages for ollama chat
        self.messages = []

        self.cross_encoder = None
        self.embedding_model = None

        if rerank_method == "cross-encoder":
            try:
                logger.info("Loading cross-encoder model BAAI/bge-reranker-v2-m3...")
                self.cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3')
                logger.info("Cross-encoder model loaded successfully")
            except Exception:
                logger.info("Loading fallback cross-encoder model BAAI/bge-reranker-base...")
                self.cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
                logger.info("Fallback cross-encoder model loaded successfully")
        elif rerank_method == "cosine":
            logger.info("Loading embedding model all-MiniLM-L6-v2...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")

        self.client = chromadb.PersistentClient(
            path=self.chroma_db_path,
            settings=Settings(allow_reset=False)
        )
        
        # Initialize CLIP model for consistent embeddings with the database
        logger.info("Loading CLIP model for embedding compatibility...")
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        logger.info("CLIP model loaded successfully")
        
        # Create CLIP embedding function to match the one used during data creation
        from chromadb.utils import embedding_functions
        
        class CLIPEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self, clip_model):
                self.clip_model = clip_model
                
            def __call__(self, input):
                """Generate embeddings using CLIP model"""
                embeddings = []
                for text in input:
                    embedding = self.clip_model.encode(text)
                    embeddings.append(embedding.tolist())
                return embeddings
        
        self.clip_ef = CLIPEmbeddingFunction(self.clip_model)
        self.collection = self.client.get_collection(self.collection_name, embedding_function=self.clip_ef)
        
        # Initialize Ollama client
        self.ollama_client = ollama.Client(host=self.ollama_url)

    def _cosine_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        return np.dot(doc_norms, query_norm)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # Search only text documents (filter out image documents for now)
        results = self.collection.query(
            query_texts=[query], 
            n_results=top_k,
            where={"content_type": "text"}  # Only search text documents
        )
        docs = [
            {
                "content": doc,
                "title": meta["title"],
                "distance": dist,
                "content_type": meta.get("content_type", "text")
            }
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ]

        if self.cross_encoder:
            pairs = [[query, d["content"][:300]] for d in docs]
            scores = self.cross_encoder.predict(pairs)
            for doc, score in zip(docs, scores):
                doc["score"] = float(score)
            return sorted(docs, key=lambda d: d["score"], reverse=True)

        elif self.embedding_model:
            query_emb = self.embedding_model.encode([query])[0]
            doc_embs = self.embedding_model.encode([d["content"][:300] for d in docs])
            sims = self._cosine_similarity(query_emb, np.array(doc_embs))
            for doc, score in zip(docs, sims):
                doc["score"] = float(score)
            return sorted(docs, key=lambda d: d["score"], reverse=True)

        return sorted(docs, key=lambda d: d["distance"])

    def build_prompt(self, query: str, docs: List[Dict[str, Any]]) -> tuple[str, str]:
        system_prompt = get_system_prompt(self.thinking_mode)
        char_limit = 800
        context_text = "\n\n".join(
            f"=== {doc['title']} ===\n{doc['content'][:char_limit]}" for doc in docs
        )
        user_prompt = get_user_prompt(context_text, query)
        return system_prompt, user_prompt

    def ask(self, query: str, max_tokens: int = 2048, top_k: int = 10, top_p: float = 0.8, temperature: float = 0.3, 
            model: str = os.getenv("DEFAULT_MODEL")) -> str:
        """
        Ask a question using the specified model (defaults to $DEFAULT_MODEL if not provided).
        """

        if not model:
            raise ValueError("model must be provided or set in DEFAULT_MODEL environment variable.")

        docs = self.search(query, top_k=top_k)
        system_prompt, user_prompt = self.build_prompt(query, docs)

        # Build messages for chat
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Add conversation history
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            messages.append({"role": "user", "content": exchange["question"]})
            messages.append({"role": "assistant", "content": exchange["answer"]})

        # Add current query
        messages.append({"role": "user", "content": user_prompt})

        full_response = ""

        if self.enable_streaming:
            for chunk in self.ollama_client.chat(
                model=model,
                messages=messages,
                stream=True,
                options={"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    full_response += chunk['message']['content']
        else:
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

    def _clean_response(self, answer: str) -> str:
        clean = re.sub(r"\033\[[0-9;]*m", "", answer).replace("LCARS: ", "").strip()
        while "<think>" in clean and "</think>" in clean:
            start = clean.find("<think>")
            end = clean.find("</think>") + len("</think>")
            clean = clean[:start] + clean[end:]
        return clean.strip()

    def _replace_thinking_tags(self, answer: str) -> str:
        clean = re.sub(r"\033\[[0-9;]*m", "", answer).replace("LCARS: ", "").strip()
        while "<think>" in clean and "</think>" in clean:
            start = clean.find("<think>")
            end = clean.find("</think>") + len("</think>")
            clean = clean[:start] + self.thinking_text + clean[end:]
        return clean.strip()

    def _update_history(self, question: str, answer: str):
        self.conversation_history.append({"question": question, "answer": answer})
        self.conversation_history = self.conversation_history[-self.max_history_turns:]

    def search_image(self, image_path: str, top_k: int = 5, 
                     model: str = os.getenv("DEFAULT_IMAGE_MODEL")) -> Dict[str, Any]:
        """
        1. Generates CLIP embedding for the provided image
        2. Searches text and image records, retrieves top_k
        3. Downloads actual images for image results
        4. Numbers and formats results
        5. Passes all info to the model to guess the theme and image
        """
        from PIL import Image
        import requests
        import tempfile
        import os

        if not model:
            raise ValueError("model must be provided or set in DEFAULT_IMAGE_MODEL environment variable.")

        # 1. Generate CLIP embedding for the image
        image = Image.open(image_path).convert('RGB')
        image_embedding = self.clip_model.encode(image)
        image_embedding = image_embedding.tolist()

        # 2. Search text and image records
        text_results = self.collection.query(
            query_embeddings=[image_embedding],
            n_results=top_k,
            where={"content_type": "text"}
        )
        image_results = self.collection.query(
            query_embeddings=[image_embedding],
            n_results=top_k,
            where={"content_type": "image"}
        )

        # 3. Download actual images for image results
        downloaded_images = []
        image_docs = image_results['documents'][0]
        image_metas = image_results['metadatas'][0]
        image_urls = [meta.get('image_url') for meta in image_metas]
        for idx, url in enumerate(image_urls):
            if url:
                try:
                    resp = requests.get(url, timeout=30)
                    if resp.status_code == 200:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            tmp.write(resp.content)
                            downloaded_images.append(tmp.name)
                    else:
                        downloaded_images.append(None)
                except Exception:
                    downloaded_images.append(None)
            else:
                downloaded_images.append(None)

        # 4. Number and format results
        formatted_text = []
        for i, (doc, meta, dist) in enumerate(zip(text_results['documents'][0], text_results['metadatas'][0], text_results['distances'][0]), 1):
            formatted_text.append(f"Text Result {i}:\nTitle: {meta.get('title', 'Unknown')}\nSimilarity: {1-dist:.4f}\nContent: {doc[:300]}\n")

        formatted_images = []
        for i, (doc, meta, dist, img_path) in enumerate(zip(image_docs, image_metas, image_results['distances'][0], downloaded_images), 1):
            formatted_images.append(f"Image Result {i}:\nImage Name: {meta.get('image_name', 'Unknown')}\nSource Page: {meta.get('source_page', 'Unknown')}\nSimilarity: {1-dist:.4f}\nDescription: {doc}\nImage Path: {img_path if img_path else 'Download failed'}\n")

        # 5. Pass all info to the model
        prompt = (
            "Given the following search results, analyze what the image may contain."
            "You may identify characters, species, organizations, types, classes, or purposes only."
            "❌ Do NOT guess episode names, scenes, locations, or events under any circumstance — unless the prompt explicitly asks for them."
            "If the entity is unclear, fall back to a broader type or category (e.g., 'humanoid alien', 'military uniform')."
            "If no identification is possible, say so clearly."
        )
        prompt += "\n".join(formatted_text)
        prompt += "\n".join(formatted_images)
        prompt += "\nRespond with one or two lines about your guess of what is in the image."

        messages = [
            {"role": "system", "content": "You are an expert Star Trek analyst."},
            {"role": "user", "content": prompt}
        ]
        response = self.ollama_client.chat(
            model=model,
            messages=messages,
            stream=False
        )
        answer = response['message']['content']

        # Clean up temp images
        for img_path in downloaded_images:
            if img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass

        return {
            "model_answer": answer
        }