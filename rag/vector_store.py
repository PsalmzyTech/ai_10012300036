"""
vector_store.py  -  PART B: Vector Storage
CS4241 - Introduction to Artificial Intelligence
Student Name:  Daniel Kingsley Bright Amusah
Index Number:  10012300036

Custom FAISS-based vector store with:
  - IndexFlatIP (inner product on L2-normalised vectors = cosine similarity)
  - Persistent save/load to disk (numpy + json)
  - Top-k retrieval with scored results
  - Metadata stored in parallel list (FAISS only stores vectors)

Design choice: IndexFlatIP over IndexIVFFlat because:
  - Our corpus is <50 k chunks — brute-force is fast enough
  - No quantisation loss — exact cosine similarity
  - Simple to implement and audit without external abstractions
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-backed vector store for chunk retrieval.

    Stores:
      - self.index  : faiss.IndexFlatIP (vectors, L2-normalised)
      - self.chunks : List[Dict]  — parallel metadata for every indexed vector
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner-product == cosine after L2-norm
        self.chunks: List[Dict[str, Any]] = []
        logger.info(f"VectorStore created (dim={dim}, IndexFlatIP)")

    # ──────────────────────────────────────────
    # ADD
    # ──────────────────────────────────────────

    def add(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """
        Add pre-computed embeddings and their corresponding chunk metadata.

        embeddings : float32 array of shape (N, dim), L2-normalised
        chunks     : list of N chunk dicts
        """
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} vectors. Total: {self.index.ntotal}")

    # ──────────────────────────────────────────
    # SEARCH
    # ──────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks for a query vector.

        Returns a list of dicts, each containing:
          - All chunk metadata fields
          - 'similarity_score' : float in [-1, 1] (cosine similarity)
          - 'rank'             : 1-based rank
        """
        if self.index.ntotal == 0:
            logger.warning("VectorStore is empty — no results returned")
            return []

        query_vec = np.array(query_vec, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:   # FAISS sentinel for "not enough results"
                continue
            result = dict(self.chunks[idx])  # copy metadata
            result["similarity_score"] = float(score)
            result["rank"] = rank
            results.append(result)

        return results

    # ──────────────────────────────────────────
    # PERSIST
    # ──────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save FAISS index + metadata to directory."""
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "faiss.index")
        meta_path  = os.path.join(directory, "chunks.json")

        faiss.write_index(self.index, index_path)

        # Save metadata (strip numpy types for JSON serialisation)
        safe_chunks = [_make_json_safe(c) for c in self.chunks]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(safe_chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"VectorStore saved to {directory}")

    def load(self, directory: str) -> None:
        """Load FAISS index + metadata from directory."""
        index_path = os.path.join(directory, "faiss.index")
        meta_path  = os.path.join(directory, "chunks.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No saved index found at {index_path}")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        logger.info(f"VectorStore loaded from {directory}: {self.index.ntotal} vectors")

    @classmethod
    def from_disk(cls, directory: str, dim: int = 384) -> "VectorStore":
        """Construct and load from disk in one call."""
        store = cls(dim)
        store.load(directory)
        return store

    def __len__(self) -> int:
        return self.index.ntotal


# ──────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────

def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
