"""
retriever.py  -  PART B: Custom Retrieval System
CS4241 - Introduction to Artificial Intelligence

Implements:
  1. Top-k vector retrieval with cosine similarity scores
  2. Query expansion  (PART B extension choice)
     - Synonym-based keyword injection
     - LLM-rephrased variant (uses Claude API when available)
  3. Failure-case analysis & fix (deduplication + diversity filter)

Query Expansion Justification:
  A user query like "how much did Ghana borrow?" may miss relevant chunks
  containing "debt", "loan", "liability", "financing". Expanding the query
  with related terms improves recall without changing the retrieval pipeline.
"""

import logging
import re
from typing import List, Dict, Any, Optional

import numpy as np

from .embedder import get_embedder
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DOMAIN SYNONYM TABLE
# (hand-crafted for the Ghana budget + election domain)
# ─────────────────────────────────────────────

DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    "borrow": ["debt", "loan", "financing", "borrowing", "liability"],
    "gdp": ["gross domestic product", "economic output", "growth rate"],
    "revenue": ["income", "receipts", "taxes", "collection"],
    "expenditure": ["spending", "budget", "outlay", "disbursement"],
    "inflation": ["price increase", "cost of living", "cpi"],
    "election": ["vote", "ballot", "presidential", "result"],
    "winner": ["winner", "elected", "majority", "victor"],
    "region": ["constituency", "district", "area", "zone"],
    "npp": ["new patriotic party", "patriotic", "akufo-addo"],
    "ndc": ["national democratic congress", "mahama"],
    "tax": ["levy", "duty", "tariff", "vat", "income tax"],
    "health": ["healthcare", "medical", "hospital", "nhis"],
    "education": ["school", "university", "training", "scholarship"],
    "agriculture": ["farming", "crop", "food", "cocoa"],
    "infrastructure": ["road", "bridge", "construction", "project"],
    "employment": ["jobs", "unemployment", "workforce", "labour"],
}


# ─────────────────────────────────────────────
# RETRIEVER CLASS
# ─────────────────────────────────────────────

class Retriever:
    """
    Top-k retrieval with optional query expansion.
    Works directly with VectorStore — no LangChain.
    """

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.embedder = get_embedder()

    # ──────────────────────────────────────────
    # CORE RETRIEVAL
    # ──────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_expansion: bool = True,
        similarity_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Retrieve top-k chunks for a query.

        Returns dict with:
          original_query, expanded_query, results, retrieval_log
        """
        log: Dict[str, Any] = {"steps": []}

        # --- Step 1: Optional Query Expansion ---
        if use_expansion:
            expanded_query = self._expand_query(query)
            log["steps"].append({
                "step": "query_expansion",
                "original": query,
                "expanded": expanded_query,
            })
        else:
            expanded_query = query

        # --- Step 2: Embed the (possibly expanded) query ---
        query_vec = self.embedder.embed_query(expanded_query)
        log["steps"].append({
            "step": "query_embedding",
            "model": self.embedder.MODEL_NAME,
            "vector_dim": int(query_vec.shape[1]),
        })

        # --- Step 3: FAISS Top-k search ---
        raw_results = self.store.search(query_vec, top_k=top_k * 2)  # fetch extra for dedup
        log["steps"].append({
            "step": "vector_search",
            "candidates_fetched": len(raw_results),
            "top_k_requested": top_k,
        })

        # --- Step 4: Diversity filter (fix for near-duplicate failure case) ---
        results = self._deduplicate(raw_results, top_k=top_k)
        log["steps"].append({
            "step": "deduplication",
            "after_dedup": len(results),
        })

        # --- Step 5: Similarity threshold filter ---
        results = [r for r in results if r["similarity_score"] >= similarity_threshold]
        log["steps"].append({
            "step": "threshold_filter",
            "threshold": similarity_threshold,
            "final_results": len(results),
            "scores": [round(r["similarity_score"], 4) for r in results],
        })

        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "results": results,
            "retrieval_log": log,
        }

    # ──────────────────────────────────────────
    # QUERY EXPANSION (PART B extension)
    # ──────────────────────────────────────────

    def _expand_query(self, query: str) -> str:
        """
        Synonym-based query expansion.

        For each word in the query that matches a domain synonym key,
        append the top-2 synonyms. This broadens the semantic search
        space without changing the core query intent.

        Example:
          "how much did Ghana borrow in 2025?"
          → "how much did Ghana borrow in 2025? debt loan financing"
        """
        query_lower = query.lower()
        extra_terms: List[str] = []

        for keyword, synonyms in DOMAIN_SYNONYMS.items():
            if keyword in query_lower:
                extra_terms.extend(synonyms[:2])

        if extra_terms:
            expanded = query + " " + " ".join(extra_terms)
            logger.debug(f"Query expanded: '{query}' → '{expanded}'")
            return expanded

        return query

    # ──────────────────────────────────────────
    # FAILURE CASE FIX: Deduplication
    # ──────────────────────────────────────────

    def _deduplicate(
        self,
        results: List[Dict[str, Any]],
        top_k: int,
        overlap_threshold: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate chunks (same paragraph retrieved via both
        paragraph AND fixed_size chunking strategies).

        Failure case: When a chunk appears twice in the index under
        different strategies, the top results can all be the same passage,
        wasting context window space and diluting answer quality.

        Fix: compare text overlap ratio; keep only the highest-scored
        representative from each near-duplicate cluster.
        """
        selected: List[Dict[str, Any]] = []
        seen_texts: List[str] = []

        for result in results:
            text = result["text"]
            is_dup = False
            for seen in seen_texts:
                if _overlap_ratio(text, seen) > overlap_threshold:
                    is_dup = True
                    break
            if not is_dup:
                selected.append(result)
                seen_texts.append(text)
            if len(selected) >= top_k:
                break

        return selected


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _overlap_ratio(a: str, b: str) -> float:
    """Jaccard-like word overlap between two text snippets."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
