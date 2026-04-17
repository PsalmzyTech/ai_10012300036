"""
pipeline.py  -  PART D: Full RAG Pipeline + Memory (PART G Innovation)
CS4241 - Introduction to Artificial Intelligence
Student Name:  Daniel Kingsley Bright Amusah
Index Number:  10012300036

Complete pipeline:
  User Query
    → [Query Expansion]
    → Embedding
    → FAISS Top-k Retrieval
    → Similarity Scoring
    → Context Selection (token budget)
    → Prompt Construction
    → LLM Generation (Claude)
    → Response + Full Log

PART G Innovation: Conversation Memory
  The pipeline maintains a rolling conversation history.
  Previous Q&A pairs are prepended as context so the LLM
  can answer follow-up questions coherently.
  Memory is stored in-session (not persisted to disk).

  This transforms a single-turn RAG into a multi-turn assistant
  without external memory frameworks.

Logging: every stage is recorded in a structured log dict
  that the UI displays alongside the response.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .chunker import chunk_documents
from .data_loader import load_all_documents
from .embedder import get_embedder
from .generator import generate_answer, generate_pure_llm
from .prompt_builder import build_prompt
from .retriever import Retriever
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

INDEX_DIR = str(Path(__file__).parent.parent / "index")
DATA_DIR = str(Path(__file__).parent.parent / "data")


# ─────────────────────────────────────────────
# INDEX BUILDER  (run once, persisted to disk)
# ─────────────────────────────────────────────

def build_index(
    data_dir: str = DATA_DIR,
    index_dir: str = INDEX_DIR,
    chunk_strategy: str = "paragraph",
    force_rebuild: bool = False,
) -> VectorStore:
    """
    Build and persist the FAISS index from source documents.
    Skips rebuild if the index already exists (unless force_rebuild=True).
    """
    index_path = os.path.join(index_dir, "faiss.index")

    if os.path.exists(index_path) and not force_rebuild:
        logger.info("Loading existing index from disk...")
        embedder = get_embedder()
        store = VectorStore.from_disk(index_dir, dim=embedder.dim)
        return store

    logger.info("Building index from scratch...")
    t0 = time.time()

    # Load raw documents
    docs = load_all_documents(data_dir)

    # Chunk documents
    chunks = chunk_documents(docs, strategy=chunk_strategy)

    # Embed chunks
    embedder = get_embedder()
    embeddings = embedder.embed_chunks(chunks, show_progress=True)

    # Build FAISS index
    store = VectorStore(dim=embedder.dim)
    store.add(embeddings, chunks)

    # Persist
    store.save(index_dir)

    elapsed = time.time() - t0
    logger.info(f"Index built in {elapsed:.1f}s: {len(chunks)} chunks indexed")
    return store


# ─────────────────────────────────────────────
# RAG PIPELINE CLASS
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline with conversation memory (Part G).

    Usage:
        pipeline = RAGPipeline()
        result = pipeline.query("What is Ghana's GDP growth target?")
    """

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        index_dir: str = INDEX_DIR,
        top_k: int = 5,
        chunk_strategy: str = "paragraph",
        prompt_template: str = "standard",
        use_expansion: bool = True,
        similarity_threshold: float = 0.0,
    ):
        self.top_k = top_k
        self.prompt_template = prompt_template
        self.use_expansion = use_expansion
        self.similarity_threshold = similarity_threshold

        # Load / build index
        self.store = build_index(data_dir, index_dir, chunk_strategy)
        self.retriever = Retriever(self.store)

        # Part G: Conversation memory
        self.memory: List[Dict[str, str]] = []  # [{"role": "user"|"assistant", "text": ...}]
        self.max_memory_turns = 5  # keep last 5 Q&A pairs

    # ──────────────────────────────────────────
    # MAIN QUERY METHOD
    # ──────────────────────────────────────────

    def query(
        self,
        user_question: str,
        template_override: Optional[str] = None,
        top_k_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a user question.

        Returns a comprehensive result dict with all intermediate outputs
        for transparency (Part D requirement).
        """
        timestamp = datetime.now().isoformat()
        t_start = time.time()

        template = template_override or self.prompt_template
        top_k = top_k_override or self.top_k

        pipeline_log: Dict[str, Any] = {
            "timestamp": timestamp,
            "user_question": user_question,
            "settings": {
                "top_k": top_k,
                "template": template,
                "use_expansion": self.use_expansion,
                "similarity_threshold": self.similarity_threshold,
            },
            "stages": {},
        }

        # ── STAGE 1: Retrieval ──────────────────
        retrieval_result = self.retriever.retrieve(
            query=user_question,
            top_k=top_k,
            use_expansion=self.use_expansion,
            similarity_threshold=self.similarity_threshold,
        )
        pipeline_log["stages"]["retrieval"] = retrieval_result["retrieval_log"]
        pipeline_log["expanded_query"] = retrieval_result["expanded_query"]
        retrieved_chunks = retrieval_result["results"]

        # ── STAGE 2: Memory injection (Part G) ──
        question_with_memory = self._inject_memory(user_question)
        pipeline_log["stages"]["memory"] = {
            "turns_in_memory": len(self.memory) // 2,
            "memory_injected": len(self.memory) > 0,
        }

        # ── STAGE 3: Prompt Construction ────────
        prompt, build_log = build_prompt(
            question=question_with_memory,
            retrieved_chunks=retrieved_chunks,
            template_name=template,
        )
        pipeline_log["stages"]["prompt_builder"] = build_log
        pipeline_log["final_prompt"] = prompt  # displayed in UI (Part D)

        # ── STAGE 4: LLM Generation ─────────────
        answer, gen_log = generate_answer(prompt)
        pipeline_log["stages"]["generation"] = gen_log

        # ── STAGE 5: Memory update ───────────────
        self._update_memory(user_question, answer)

        elapsed = time.time() - t_start
        pipeline_log["total_time_s"] = round(elapsed, 2)

        return {
            "question": user_question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "pipeline_log": pipeline_log,
        }

    # ──────────────────────────────────────────
    # PART E: PURE LLM COMPARISON
    # ──────────────────────────────────────────

    def query_pure_llm(self, question: str) -> Dict[str, Any]:
        """Run the same question through Claude with NO retrieval context."""
        answer, gen_log = generate_pure_llm(question)
        return {"question": question, "answer": answer, "gen_log": gen_log}

    # ──────────────────────────────────────────
    # PART G: MEMORY MANAGEMENT
    # ──────────────────────────────────────────

    def _inject_memory(self, question: str) -> str:
        """Prepend recent conversation history to the question."""
        if not self.memory:
            return question

        history_lines = ["Previous conversation:"]
        for turn in self.memory[-self.max_memory_turns * 2:]:
            prefix = "User" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{prefix}: {turn['text']}")

        history_lines.append(f"\nCurrent question: {question}")
        return "\n".join(history_lines)

    def _update_memory(self, question: str, answer: str) -> None:
        """Add latest Q&A pair to memory, trimming if needed."""
        self.memory.append({"role": "user", "text": question})
        self.memory.append({"role": "assistant", "text": answer})

        # Keep only recent turns
        max_items = self.max_memory_turns * 2
        if len(self.memory) > max_items:
            self.memory = self.memory[-max_items:]

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self.memory = []
        logger.info("Conversation memory cleared")

    # ──────────────────────────────────────────
    # PART G: FEEDBACK LOOP
    # ──────────────────────────────────────────

    def record_feedback(
        self,
        question: str,
        answer: str,
        rating: int,  # 1 = helpful, -1 = not helpful
        feedback_dir: Optional[str] = None,
    ) -> None:
        """
        Log user feedback for later retrieval improvement analysis.

        Part G Innovation: Feedback loop
          Stores (question, answer, rating) to a JSONL file.
          In a production system, negative feedback entries could be used to:
            1. Identify queries where retrieval failed (low similarity + negative rating)
            2. Fine-tune the embedding model on hard negatives
            3. Adjust the similarity threshold dynamically
        """
        if feedback_dir is None:
            feedback_dir = str(Path(__file__).parent.parent / "logs")

        os.makedirs(feedback_dir, exist_ok=True)
        fb_path = os.path.join(feedback_dir, "feedback.jsonl")

        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "rating": rating,  # 1 = positive, -1 = negative
        }

        with open(fb_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(f"Feedback recorded: rating={rating} for '{question[:60]}...'")
