"""
chunker.py  -  PART A: Chunking Strategies
CS4241 - Introduction to Artificial Intelligence
Student Name:  Daniel Kingsley Bright Amusah
Index Number:  10012300036

Implements THREE chunking strategies with justification:

  1. FIXED-SIZE chunking   (chunk_size=512 chars, overlap=64)
     - Predictable memory use; good baseline for PDF prose
     - Overlap prevents splitting mid-sentence context

  2. SENTENCE chunking     (N sentences per chunk, overlap=1 sentence)
     - Preserves semantic units; best for factual queries
     - Each sentence is a complete thought

  3. PARAGRAPH chunking    (split on blank lines, min/max token guard)
     - Budget PDF is structured by paragraphs; keeps thematic units together
     - Better for "what does section X say" queries

Election CSV rows are pre-converted to sentences in data_loader.py and
treated as single-sentence chunks (already minimal / atomic).

Design decision: PDF uses paragraph chunking as primary strategy because
the budget document is logically organised by topic paragraphs. Sentence
chunking is offered as an alternative for comparison.
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STRATEGY 1: Fixed-size character chunking
# ─────────────────────────────────────────────

def fixed_size_chunks(
    doc: Dict[str, Any],
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Dict[str, Any]]:
    """
    Split document text into fixed-size character windows with overlap.

    Justification:
      chunk_size=512 fits within typical embedding model token limits (~256 tokens).
      overlap=64 (~12%) prevents losing cross-boundary context.
    """
    text = doc["text"]
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk = _make_chunk(doc, chunk_text, chunk_idx, "fixed_size")
            chunks.append(chunk)
            chunk_idx += 1

        start += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────────
# STRATEGY 2: Sentence-based chunking
# ─────────────────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def sentence_chunks(
    doc: Dict[str, Any],
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
) -> List[Dict[str, Any]]:
    """
    Group N consecutive sentences into a chunk with sentence-level overlap.

    Justification:
      sentences_per_chunk=5 gives ~150-250 tokens — enough context without
      over-filling the embedding model window.
      overlap_sentences=1 stitches boundaries without doubling content.
    """
    text = doc["text"]
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    chunks = []
    chunk_idx = 0
    step = max(1, sentences_per_chunk - overlap_sentences)

    for i in range(0, len(sentences), step):
        window = sentences[i : i + sentences_per_chunk]
        chunk_text = " ".join(window).strip()
        if chunk_text:
            chunk = _make_chunk(doc, chunk_text, chunk_idx, "sentence")
            chunks.append(chunk)
            chunk_idx += 1

    return chunks


# ─────────────────────────────────────────────
# STRATEGY 3: Paragraph-based chunking (PRIMARY)
# ─────────────────────────────────────────────

def paragraph_chunks(
    doc: Dict[str, Any],
    min_chars: int = 100,
    max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    """
    Split on blank lines; merge short paragraphs and split long ones.

    Justification:
      The Budget PDF uses blank-line-delimited paragraphs as semantic units.
      min_chars=100 avoids near-empty chunks (e.g. section headers alone).
      max_chars=1200 keeps chunks within embedding model limits.
      This strategy preserves topical coherence better than character windows.
    """
    text = doc["text"]
    raw_paragraphs = re.split(r"\n\s*\n", text)

    merged: List[str] = []
    buffer = ""

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(buffer) + len(para) < min_chars:
            buffer = (buffer + " " + para).strip()
        else:
            if buffer:
                merged.append(buffer)
            buffer = para

    if buffer:
        merged.append(buffer)

    chunks = []
    chunk_idx = 0

    for para in merged:
        if len(para) <= max_chars:
            chunk = _make_chunk(doc, para, chunk_idx, "paragraph")
            chunks.append(chunk)
            chunk_idx += 1
        else:
            # Long paragraph: fall back to fixed-size sub-chunking
            sub_doc = dict(doc, text=para)
            sub_chunks = fixed_size_chunks(sub_doc, chunk_size=max_chars, overlap=100)
            for sc in sub_chunks:
                sc["chunk_strategy"] = "paragraph+fixed_fallback"
                sc["chunk_idx"] = chunk_idx
                chunks.append(sc)
                chunk_idx += 1

    return chunks


# ─────────────────────────────────────────────
# DISPATCHER
# ─────────────────────────────────────────────

def chunk_documents(
    docs: List[Dict[str, Any]],
    strategy: str = "paragraph",
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Chunk a list of documents using the chosen strategy.

    strategy choices: 'paragraph' (default), 'sentence', 'fixed_size'

    Election CSV rows (doc_type='election_csv') are already atomic —
    they are returned unchanged regardless of strategy.
    """
    all_chunks: List[Dict[str, Any]] = []
    strategy_fn = {
        "paragraph": paragraph_chunks,
        "sentence": sentence_chunks,
        "fixed_size": fixed_size_chunks,
    }.get(strategy)

    if strategy_fn is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: paragraph, sentence, fixed_size")

    for doc in docs:
        if doc.get("doc_type") == "election_csv":
            # Already sentence-level; treat as one chunk
            chunk = _make_chunk(doc, doc["text"], 0, "csv_row")
            all_chunks.append(chunk)
        else:
            chunks = strategy_fn(doc, **kwargs)
            all_chunks.extend(chunks)

    logger.info(f"Chunking ({strategy}): {len(docs)} docs -> {len(all_chunks)} chunks")
    return all_chunks


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _make_chunk(
    doc: Dict[str, Any],
    text: str,
    idx: int,
    strategy: str,
) -> Dict[str, Any]:
    """Create a standardised chunk dict from a document and text slice."""
    chunk = {k: v for k, v in doc.items()}  # copy metadata
    chunk["text"] = text
    chunk["chunk_idx"] = idx
    chunk["chunk_strategy"] = strategy
    chunk["char_count"] = len(text)
    return chunk
