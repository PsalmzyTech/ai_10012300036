"""
embedder.py  -  PART B: Embedding Pipeline
CS4241 - Introduction to Artificial Intelligence
Student Name:  Daniel Kingsley Bright Amusah
Index Number:  10012300036

Implements a custom embedding pipeline using sentence-transformers.
Model: all-MiniLM-L6-v2
  - 384-dimensional dense vectors
  - Trained for semantic similarity (ideal for RAG retrieval)
  - Runs locally — no API key required
  - Fast: ~14k sentences/second on CPU

No LangChain or LlamaIndex wrappers used.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Union

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# EMBEDDER CLASS
# ─────────────────────────────────────────────

class Embedder:
    """
    Wraps sentence-transformers to produce L2-normalised embeddings
    suitable for cosine similarity via dot product.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        logger.info(f"Loading embedding model: {self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dim}")

    # ------------------------------------------------------------------
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of strings.
        Returns float32 array of shape (len(texts), dim), L2-normalised.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine sim = dot product after L2 norm
        )
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        Returns float32 array of shape (1, dim), L2-normalised.
        """
        vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.astype(np.float32)

    # ------------------------------------------------------------------
    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Convenience wrapper: embed a list of chunk dicts (reads 'text' key).
        Returns float32 array of shape (len(chunks), dim).
        """
        texts = [c["text"] for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embed_texts(texts, batch_size=batch_size, show_progress=show_progress)
        logger.info("Embedding complete.")
        return embeddings


# ─────────────────────────────────────────────
# SINGLETON (lazy-loaded, shared across modules)
# ─────────────────────────────────────────────

_embedder_instance: Union[Embedder, None] = None


def get_embedder() -> Embedder:
    """Return the shared Embedder instance (loaded once)."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
