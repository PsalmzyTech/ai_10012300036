"""
generator.py  -  LLM Response Generation
CS4241 - Introduction to Artificial Intelligence
Student Name:  Daniel Kingsley Bright Amusah
Index Number:  10012300036

Wraps the Anthropic Claude API for response generation.
Model: claude-haiku-4-5-20251001 (fast, cost-effective for RAG answers)

Also provides a "no-retrieval" mode for Part E comparison:
  generate_pure_llm() — same question, no context injected.
"""

import logging
import os
from typing import Optional, Dict, Any, Tuple

import anthropic

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CLAUDE CLIENT
# ─────────────────────────────────────────────

def _get_client() -> anthropic.Anthropic:
    """Create an Anthropic client using ANTHROPIC_API_KEY env var."""
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. "
            "Add it to your .env file or environment variables."
        )
    return anthropic.Anthropic(api_key=api_key)


# ─────────────────────────────────────────────
# RAG GENERATION (with context)
# ─────────────────────────────────────────────

def generate_answer(
    prompt: str,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> Tuple[str, Dict[str, Any]]:
    """
    Send the RAG prompt to Claude and return the response.

    Low temperature (0.2) reduces hallucination for factual retrieval tasks.

    Returns (answer_text, generation_log).
    """
    client = _get_client()

    logger.info(f"Calling LLM: model={model}, max_tokens={max_tokens}, temp={temperature}")

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = message.content[0].text

    gen_log = {
        "model": model,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "stop_reason": message.stop_reason,
        "temperature": temperature,
    }

    logger.info(
        f"LLM response: {gen_log['output_tokens']} output tokens, "
        f"stop={gen_log['stop_reason']}"
    )

    return answer, gen_log


# ─────────────────────────────────────────────
# PURE LLM (no retrieval) — Part E comparison
# ─────────────────────────────────────────────

PURE_LLM_SYSTEM = """\
You are a knowledgeable assistant. Answer the user's question based on your \
general knowledge. Be factual and concise."""


def generate_pure_llm(
    question: str,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a response WITHOUT any retrieved context.
    Used in Part E for RAG vs. pure-LLM comparison.

    Higher temperature (0.7) used here because there is no ground-truth
    context — we want to observe potential hallucination behaviour.
    """
    client = _get_client()

    logger.info(f"Pure LLM call (no retrieval): model={model}")

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=PURE_LLM_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )

    answer = message.content[0].text

    gen_log = {
        "mode": "pure_llm",
        "model": model,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "stop_reason": message.stop_reason,
        "temperature": temperature,
    }

    return answer, gen_log
