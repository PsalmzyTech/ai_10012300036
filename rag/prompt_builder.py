"""
prompt_builder.py  -  PART C: Prompt Engineering & Generation
CS4241 - Introduction to Artificial Intelligence
Student Name:  Daniel Kingsley Bright Amusah
Index Number:  10012300036

Implements:
  1. Structured prompt template with context injection
  2. Hallucination control instructions
  3. Context window management (token budget enforcement)
  4. Multiple prompt variants for experiment comparison

Design decisions:
  - System prompt separates role, instructions, and context clearly
  - "Only use the provided context" guards against hallucination
  - Token budget enforced by character proxy (1 token ≈ 4 chars)
  - Three template variants to compare in experiments
"""

import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Approximate chars per token (rough proxy, avoids tiktoken dependency)
CHARS_PER_TOKEN = 4
# Max tokens reserved for context (leave room for system prompt + response)
CONTEXT_TOKEN_BUDGET = 3000
CONTEXT_CHAR_BUDGET = CONTEXT_TOKEN_BUDGET * CHARS_PER_TOKEN


# ─────────────────────────────────────────────
# CONTEXT WINDOW MANAGEMENT
# ─────────────────────────────────────────────

def select_context_chunks(
    retrieved: List[Dict[str, Any]],
    char_budget: int = CONTEXT_CHAR_BUDGET,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Greedily select top chunks that fit within the character budget.

    Returns (selected_chunks, dropped_chunks).
    Chunks are already ranked by similarity score; we include
    the highest-scored ones first.
    """
    selected: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    used = 0

    for chunk in retrieved:
        chunk_len = len(chunk["text"])
        if used + chunk_len <= char_budget:
            selected.append(chunk)
            used += chunk_len
        else:
            dropped.append(chunk)

    logger.debug(
        f"Context selection: {len(selected)} chunks ({used} chars) "
        f"| {len(dropped)} dropped"
    )
    return selected, dropped


def format_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Format selected chunks into a numbered context block for injection.
    """
    lines = []
    for i, c in enumerate(chunks, start=1):
        score = c.get("similarity_score", 0.0)
        source = c.get("source", "unknown")
        lines.append(f"[Chunk {i} | Score: {score:.4f} | Source: {source}]")
        lines.append(c["text"].strip())
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────

# --- Template A: Standard RAG (baseline) ---
TEMPLATE_A = """\
You are an AI assistant for Academic City University, specialising in Ghana's \
2025 Budget Statement and Ghana presidential election results.

INSTRUCTIONS:
- Answer ONLY using the provided context below.
- If the answer is not found in the context, say exactly: \
"I don't have enough information in the provided documents to answer that."
- Do NOT invent facts, figures, or statistics.
- Cite the source (Chunk number) for each key claim.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""


# --- Template B: Chain-of-thought reasoning (PART G innovation) ---
TEMPLATE_B = """\
You are an AI assistant for Academic City University. \
Your knowledge is strictly limited to the context provided.

INSTRUCTIONS:
1. Read the context carefully.
2. Think step-by-step before answering.
3. If information is missing, explicitly state what is missing.
4. Cite chunk numbers for each claim.
5. Do NOT hallucinate — if the context does not confirm a fact, say so.

CONTEXT:
{context}

USER QUESTION:
{question}

REASONING:
Let me work through the context step by step.

ANSWER:"""


# --- Template C: Strict factual mode ---
TEMPLATE_C = """\
You are a precise factual assistant. \
Answer the question using ONLY the numbered chunks below. \
Do not add information from general knowledge.

{context}

Q: {question}
A (cite chunk numbers):"""


TEMPLATES = {
    "standard": TEMPLATE_A,
    "chain_of_thought": TEMPLATE_B,
    "strict_factual": TEMPLATE_C,
}


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

def build_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    template_name: str = "standard",
    char_budget: int = CONTEXT_CHAR_BUDGET,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build the final prompt string to send to the LLM.

    Returns:
      (prompt_str, build_log) where build_log records:
        - template used
        - chunks selected / dropped
        - final prompt character count
    """
    template = TEMPLATES.get(template_name, TEMPLATE_A)

    # --- Context window management ---
    selected, dropped = select_context_chunks(retrieved_chunks, char_budget)
    context_block = format_context_block(selected)

    prompt = template.format(context=context_block, question=question)

    build_log = {
        "template": template_name,
        "chunks_selected": len(selected),
        "chunks_dropped": len(dropped),
        "context_chars": len(context_block),
        "total_prompt_chars": len(prompt),
        "estimated_tokens": len(prompt) // CHARS_PER_TOKEN,
    }

    logger.info(
        f"Prompt built: template={template_name}, "
        f"chunks={len(selected)}, ~{build_log['estimated_tokens']} tokens"
    )

    return prompt, build_log
