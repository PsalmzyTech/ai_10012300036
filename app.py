"""
app.py  -  FINAL DELIVERABLE: Streamlit UI
CS4241 - Introduction to Artificial Intelligence
Academic City University

RAG Chatbot for Ghana 2025 Budget Statement & Election Results

Features (Part D requirements):
  - Query input
  - Display retrieved chunks with similarity scores
  - Show final prompt sent to LLM
  - Show LLM response

Part G extras:
  - Conversation memory (multi-turn chat)
  - Feedback buttons (thumbs up / down)
  - Adversarial test panel (Part E)
  - Prompt template switcher (Part C experiments)
"""

import os
import sys
import base64
import logging
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# ─── Load environment variables ───────────────
load_dotenv()  # loads .env locally

# Support Streamlit Cloud secrets (st.secrets) as well
try:
    import streamlit as _st_check
    if hasattr(_st_check, "secrets") and "ANTHROPIC_API_KEY" in _st_check.secrets:
        os.environ["ANTHROPIC_API_KEY"] = _st_check.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

# ─── Configure logging ────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─── Page config ──────────────────────────────
st.set_page_config(
    page_title="ACity RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Blurred campus background ───────────────
def set_bg_image(image_path: str, blur: int = 6, brightness: float = 0.45):
    """Inject CSS to use a local image as a blurred, darkened full-page background."""
    img_path = Path(image_path)
    if not img_path.exists():
        return
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    ext = img_path.suffix.lower().replace(".", "")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    st.markdown(f"""
    <style>
    /* ── full-page background ── */
    .stApp {{
        background-image: url("data:image/{mime};base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    /* blur + darken overlay */
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        backdrop-filter: blur({blur}px);
        -webkit-backdrop-filter: blur({blur}px);
        background: rgba(0, 0, 0, {1 - brightness});
        z-index: 0;
    }}
    /* ensure all content sits above the overlay */
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}
    /* glass-card effect for main content blocks */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"],
    [data-testid="stChatMessageContent"],
    .stTabs [data-baseweb="tab-panel"] {{
        background: rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 0.5rem;
    }}
    /* sidebar semi-transparent */
    [data-testid="stSidebar"] {{
        background: rgba(0, 20, 60, 0.75) !important;
        backdrop-filter: blur(10px);
    }}
    [data-testid="stSidebar"] * {{
        color: #e8eaf6 !important;
    }}
    /* headings bright white */
    h1, h2, h3, h4 {{
        color: #ffffff !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.7);
    }}
    /* chat bubbles */
    [data-testid="stChatMessageContent"] {{
        background: rgba(255, 255, 255, 0.12) !important;
        border: 1px solid rgba(255,255,255,0.18);
        color: #f0f0f0 !important;
    }}
    /* input box */
    [data-testid="stChatInputTextArea"] {{
        background: rgba(255,255,255,0.15) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }}
    /* tab labels */
    button[data-baseweb="tab"] {{
        color: #cfd8dc !important;
        font-weight: 600;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #ffffff !important;
        border-bottom: 3px solid #42a5f5 !important;
    }}
    /* metric text */
    [data-testid="stMetricValue"] {{
        color: #80cbc4 !important;
    }}
    /* general text */
    p, li, label, span {{
        color: #e0e0e0 !important;
    }}
    /* expander headers */
    [data-testid="stExpander"] summary {{
        color: #b3e5fc !important;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg_image("assets/campus.jpg", blur=7, brightness=0.45)

# ─── Lazy-load the pipeline ───────────────────
@st.cache_resource(show_spinner="Building knowledge index (first run may take ~2 min)...")
def load_pipeline():
    from rag.pipeline import RAGPipeline
    return RAGPipeline()


# ─────────────────────────────────────────────
# SIDEBAR SETTINGS
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Academic_City_University_College_Ghana_Logo.png/200px-Academic_City_University_College_Ghana_Logo.png",
             use_column_width=True)

    st.title("RAG Settings")

    top_k = st.slider("Top-K Retrieved Chunks", 1, 10, 5)
    template = st.selectbox(
        "Prompt Template",
        ["standard", "chain_of_thought", "strict_factual"],
        index=0,
        help=(
            "standard: balanced RAG prompt\n"
            "chain_of_thought: step-by-step reasoning\n"
            "strict_factual: minimal, citation-heavy"
        ),
    )
    use_expansion = st.checkbox("Query Expansion", value=True,
                                help="Expand query with domain synonyms for better recall")
    sim_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.0, 0.05,
                               help="Minimum cosine similarity to include a chunk")

    st.divider()
    show_prompt = st.checkbox("Show Final Prompt", value=False)
    show_log    = st.checkbox("Show Pipeline Log", value=False)

    st.divider()
    if st.button("Clear Conversation Memory"):
        if "pipeline" in st.session_state:
            st.session_state.pipeline.clear_memory()
        st.session_state.messages = []
        st.session_state.last_result = None
        st.success("Memory cleared!")

    st.divider()
    st.caption("CS4241 – Introduction to Artificial Intelligence")
    st.caption("Academic City University • 2026")


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────

st.title("🎓 Academic City RAG Chatbot")
st.markdown(
    "Chat with Ghana's **2025 Budget Statement** and **Presidential Election Results** "
    "using a custom RAG pipeline (no LangChain / LlamaIndex)."
)

tab_chat, tab_eval, tab_arch = st.tabs(["💬 Chat", "🧪 Evaluation (Part E)", "🏗️ Architecture (Part F)"])


# ─────────────────────────────────────────────
# TAB 1: CHAT
# ─────────────────────────────────────────────

with tab_chat:
    # Initialise session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask about Ghana's 2025 budget or election results...")

    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Load pipeline (cached)
        if "pipeline" not in st.session_state:
            st.session_state.pipeline = load_pipeline()

        pipeline = st.session_state.pipeline
        pipeline.top_k = top_k
        pipeline.prompt_template = template
        pipeline.use_expansion = use_expansion
        pipeline.similarity_threshold = sim_threshold

        # Run the RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                result = pipeline.query(user_input)
                st.session_state.last_result = result

            answer = result["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # ── Retrieved Chunks Display (Part D requirement) ──
            with st.expander(f"📚 Retrieved Chunks ({len(result['retrieved_chunks'])})", expanded=False):
                for chunk in result["retrieved_chunks"]:
                    score = chunk.get("similarity_score", 0)
                    source = chunk.get("source", "unknown")
                    strategy = chunk.get("chunk_strategy", "")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Source:** `{source}`  |  **Strategy:** `{strategy}`")
                        st.text(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))
                    with col2:
                        st.metric("Similarity", f"{score:.4f}")
                    st.divider()

            # ── Final Prompt Display (optional, Part D) ──
            if show_prompt:
                with st.expander("🔤 Final Prompt Sent to LLM", expanded=False):
                    st.code(result["pipeline_log"]["final_prompt"], language="text")

            # ── Pipeline Log (optional) ──
            if show_log:
                with st.expander("📋 Pipeline Log", expanded=False):
                    import json
                    log = result["pipeline_log"]
                    st.json({
                        "expanded_query": log.get("expanded_query"),
                        "retrieval": log["stages"].get("retrieval"),
                        "prompt_builder": log["stages"].get("prompt_builder"),
                        "generation": log["stages"].get("generation"),
                        "total_time_s": log.get("total_time_s"),
                    })

            # ── Feedback buttons (Part G) ──
            st.markdown("**Was this answer helpful?**")
            fcol1, fcol2, _ = st.columns([1, 1, 8])
            with fcol1:
                if st.button("👍", key=f"up_{len(st.session_state.messages)}"):
                    pipeline.record_feedback(user_input, answer, rating=1)
                    st.success("Thanks for your feedback!")
            with fcol2:
                if st.button("👎", key=f"dn_{len(st.session_state.messages)}"):
                    pipeline.record_feedback(user_input, answer, rating=-1)
                    st.info("Thanks — we'll improve retrieval for this query.")


# ─────────────────────────────────────────────
# TAB 2: EVALUATION (Part E)
# ─────────────────────────────────────────────

with tab_eval:
    st.header("Part E: Critical Evaluation & Adversarial Testing")

    st.markdown("""
    **Adversarial queries** test the system's ability to handle:
    - Ambiguous or vague questions
    - Misleading / out-of-scope queries
    - Questions that have no answer in the documents
    """)

    ADVERSARIAL_QUERIES = [
        "Who won the 2025 elections?",       # Ambiguous: no 2025 election data
        "What is Ghana's total GDP in dollars?",  # Misleading: data may be in GHS/growth %
        "Compare Ghana's election results with Nigeria's",  # Out-of-scope
        "What was the inflation rate last month?",  # No current data in static docs
    ]

    selected_q = st.selectbox("Pick an adversarial query:", ADVERSARIAL_QUERIES)
    custom_q   = st.text_input("Or type your own adversarial query:")
    test_query = custom_q if custom_q.strip() else selected_q

    if st.button("Run Evaluation", type="primary"):
        if "pipeline" not in st.session_state:
            st.session_state.pipeline = load_pipeline()
        pipeline = st.session_state.pipeline

        with st.spinner("Running RAG + Pure LLM comparison..."):
            rag_result  = pipeline.query(test_query)
            pure_result = pipeline.query_pure_llm(test_query)

        col_rag, col_pure = st.columns(2)

        with col_rag:
            st.subheader("RAG System Answer")
            st.success(rag_result["answer"])
            st.markdown("**Retrieved chunks:**")
            for c in rag_result["retrieved_chunks"]:
                st.caption(f"Score {c['similarity_score']:.4f} | {c['source'][:60]}")

        with col_pure:
            st.subheader("Pure LLM Answer (No Retrieval)")
            st.warning(pure_result["answer"])

        st.divider()
        st.subheader("Analysis")
        st.markdown("""
        | Criterion | RAG System | Pure LLM |
        |-----------|-----------|----------|
        | Grounded in documents | ✅ Yes | ❌ No |
        | Cites sources | ✅ Yes | ❌ No |
        | Hallucination risk | Low (context-bound) | Higher (general knowledge) |
        | Handles out-of-scope | Gracefully declines | May fabricate |
        """)


# ─────────────────────────────────────────────
# TAB 3: ARCHITECTURE (Part F)
# ─────────────────────────────────────────────

with tab_arch:
    st.header("Part F: System Architecture")

    st.markdown("""
    ## RAG System Architecture

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    DATA INGESTION (Part A)                       │
    │                                                                   │
    │  PDF (252 pages)  ──► pdfplumber extraction ──► text pages       │
    │  CSV (elections)  ──► pandas cleaning       ──► sentence rows     │
    │                                ▼                                  │
    │            Chunker (paragraph / sentence / fixed-size)           │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │ chunks
    ┌──────────────────────────────▼──────────────────────────────────┐
    │                  EMBEDDING PIPELINE (Part B)                     │
    │                                                                   │
    │         SentenceTransformer (all-MiniLM-L6-v2, 384-dim)         │
    │              L2-normalised float32 vectors                       │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │ embeddings
    ┌──────────────────────────────▼──────────────────────────────────┐
    │                  VECTOR STORE (Part B)                           │
    │                                                                   │
    │         FAISS IndexFlatIP  (exact cosine similarity)             │
    │         Metadata: chunk text, source, doc_type, scores           │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │ saved to disk
    ╔══════════════════════════════▼══════════════════════════════════╗
    ║              QUERY TIME  (Parts B, C, D)                        ║
    ║                                                                   ║
    ║  User Query                                                       ║
    ║      │                                                            ║
    ║      ▼                                                            ║
    ║  Query Expansion  (synonym injection — Part B extension)         ║
    ║      │                                                            ║
    ║      ▼                                                            ║
    ║  Embed Query  (same SentenceTransformer model)                   ║
    ║      │                                                            ║
    ║      ▼                                                            ║
    ║  FAISS Top-k Search + Deduplication filter                       ║
    ║      │                                                            ║
    ║      ▼                                                            ║
    ║  Context Selection  (token budget, rank-ordered)                 ║
    ║      │                                                            ║
    ║      ▼                                                            ║
    ║  Prompt Builder  (template + context injection, Part C)          ║
    ║      │                                                            ║
    ║      ▼                                                            ║
    ║  Claude LLM  (claude-haiku-4-5, temp=0.2)                        ║
    ║      │                                                            ║
    ║      ▼                                                            ║
    ║  Response + Full Pipeline Log displayed in Streamlit UI          ║
    ║      │                                                            ║
    ║      ▼                                                            ║
    ║  Conversation Memory update  (Part G innovation)                 ║
    ║      │                                                            ║
    ║      ▼                                                            ║
    ║  Feedback logging  (thumbs up/down → feedback.jsonl)             ║
    ╚═════════════════════════════════════════════════════════════════╝
    ```

    ## Component Justification

    | Component | Choice | Reason |
    |-----------|--------|--------|
    | Embedding model | all-MiniLM-L6-v2 | Fast, accurate, runs locally, 384-dim fits FAISS well |
    | Vector index | FAISS IndexFlatIP | Exact cosine sim; corpus < 50k chunks so brute-force is fast |
    | Chunking | Paragraph (primary) | Budget doc is paragraph-structured; preserves thematic coherence |
    | LLM | Claude Haiku 4.5 | Fast, cost-effective, strong instruction following |
    | Temperature | 0.2 (RAG mode) | Low temp reduces hallucination; context is already precise |
    | Memory | In-session rolling list | Lightweight; no external DB needed; supports follow-up questions |
    | Feedback | JSONL log | Simple, portable, analyzable; basis for future fine-tuning |

    ## Why this design suits the domain

    The Ghana Budget PDF is a formal government document (~252 pages) organised
    by thematic chapters. Paragraph-level chunking aligns with this structure,
    keeping policy context intact. The election CSV is tabular — converting rows
    to natural-language sentences allows the same embedding/retrieval pipeline
    to work uniformly across both data types without special-casing.
    """)
