# System Architecture Document
## CS4241 – Introduction to Artificial Intelligence
### RAG Chatbot — Part F: Architecture & System Design

---

## 1. Overview

This system is a custom Retrieval-Augmented Generation (RAG) chatbot built without
any end-to-end RAG frameworks (no LangChain, LlamaIndex, etc.). All core components
are manually implemented in Python.

The chatbot answers questions about:
- **Ghana's 2025 Budget Statement and Economic Policy** (252-page PDF)
- **Ghana Presidential Election Results** (CSV dataset, 2020 & 2024)

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                 │
│                                                                       │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐   │
│  │  2025 Budget PDF     │    │  Ghana Election Results CSV       │   │
│  │  (252 pages, 4 MB)   │    │  (Year, Region, Candidate,       │   │
│  │  mofep.gov.gh        │    │   Party, Votes, Votes%)           │   │
│  └──────────┬───────────┘    └──────────────────┬───────────────┘   │
│             │ pdfplumber                         │ pandas             │
│             ▼                                    ▼                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              data_loader.py  (PART A)                        │   │
│  │  • Page-level text extraction + cleaning                     │   │
│  │  • CSV row → natural-language sentence conversion            │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
└─────────────────────────────────┼───────────────────────────────────┘
                                  │ List[Dict] — raw documents
┌─────────────────────────────────▼───────────────────────────────────┐
│                    CHUNKING PIPELINE (PART A)                        │
│                                                                       │
│  chunker.py                                                          │
│                                                                       │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐ │
│  │ Paragraph Chunks│  │ Sentence Chunks  │  │ Fixed-Size Chunks  │ │
│  │ (default)       │  │ (5 sent, 1 olap) │  │ (512 chars, 64     │ │
│  │ 100–1200 chars  │  │                  │  │  char overlap)     │ │
│  └────────┬────────┘  └──────────────────┘  └────────────────────┘ │
│           │ (primary strategy)                                        │
│           ▼                                                           │
│  ~2,300 PDF chunks + ~1,600 CSV row chunks = ~3,900 total chunks     │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ List[Dict] — chunks
┌─────────────────────────────────▼───────────────────────────────────┐
│                   EMBEDDING PIPELINE (PART B)                        │
│                                                                       │
│  embedder.py                                                         │
│                                                                       │
│  Model: sentence-transformers/all-MiniLM-L6-v2                       │
│  Output: float32 vectors, shape (N, 384), L2-normalised              │
│  Runs locally (no API key required for indexing)                     │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ np.ndarray (N, 384)
┌─────────────────────────────────▼───────────────────────────────────┐
│                    VECTOR STORE (PART B)                             │
│                                                                       │
│  vector_store.py                                                     │
│                                                                       │
│  Index type : faiss.IndexFlatIP                                      │
│  Similarity : Cosine (= inner product after L2 normalisation)        │
│  Persistence: faiss.index + chunks.json saved to ./index/            │
│  Metadata   : source, page_num, doc_type, chunk_strategy, text       │
└────────────────────────────── OFFLINE ──────────────────────────────┘
                                  (built once, reused every query)

══════════════════════════  QUERY TIME  ══════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (Streamlit)                       │
│                                                                       │
│  app.py  — 3 tabs: Chat | Evaluation | Architecture                  │
│  Sidebar: top_k, template, expansion toggle, threshold slider        │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ user_question (string)
┌─────────────────────────────────▼───────────────────────────────────┐
│                    RAG PIPELINE (PART D)                             │
│                                                                       │
│  pipeline.py — RAGPipeline.query()                                   │
│                                                                       │
│  ① Memory Injection (Part G)                                        │
│     └─ Prepend last N Q&A turns to contextualise follow-ups         │
│                                                                       │
│  ② Query Expansion (Part B extension)                               │
│     └─ retriever.py — synonym table lookup                          │
│        "borrow" → +debt, +loan, +financing                          │
│                                                                       │
│  ③ Query Embedding                                                  │
│     └─ Same SentenceTransformer model (all-MiniLM-L6-v2)           │
│        Output: (1, 384) float32 vector                               │
│                                                                       │
│  ④ FAISS Top-k Search                                               │
│     └─ Returns top 2k candidates with cosine similarity scores      │
│                                                                       │
│  ⑤ Deduplication Filter                                            │
│     └─ Jaccard word overlap → drops near-duplicate chunks           │
│                                                                       │
│  ⑥ Similarity Threshold Filter                                      │
│     └─ Configurable minimum score (default 0.0)                     │
│                                                                       │
│  ⑦ Context Selection                                               │
│     └─ prompt_builder.py — greedy token budget (3000 tokens)        │
│        Highest-scored chunks fill the budget first                   │
│                                                                       │
│  ⑧ Prompt Construction (Part C)                                    │
│     └─ Template: standard | chain_of_thought | strict_factual       │
│        Injects: role + instructions + context + question             │
│        Hallucination guard: "only use provided context"              │
│                                                                       │
│  ⑨ LLM Generation                                                  │
│     └─ generator.py — Claude claude-haiku-4-5-20251001             │
│        temperature=0.2 (low → factual, less hallucination)          │
│        max_tokens=1024                                                │
│                                                                       │
│  ⑩ Response + Full Log returned to UI                              │
│                                                                       │
│  ⑪ Memory Update                                                   │
│     └─ Append Q&A pair to rolling in-memory list                    │
│                                                                       │
│  ⑫ Feedback Logging (Part G)                                       │
│     └─ Thumbs up/down → logs/feedback.jsonl                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

```
User Question (string)
    │
    ├─► [Query Expansion] add domain synonyms
    │
    ├─► [Embed] SentenceTransformer → 384-dim vector
    │
    ├─► [FAISS Search] top-2k candidates + cosine scores
    │
    ├─► [Filter] dedup + threshold → top-k chunks
    │
    ├─► [Prompt] template.format(context, question)
    │
    ├─► [LLM] Claude Haiku API call
    │
    └─► Response → UI display + memory + feedback log
```

---

## 4. Component Interaction Table

| Module | Depends On | Called By |
|--------|-----------|-----------|
| `data_loader.py` | pdfplumber, pandas | `pipeline.py` (build_index) |
| `chunker.py` | data_loader output | `pipeline.py` (build_index) |
| `embedder.py` | sentence-transformers | `pipeline.py`, `retriever.py` |
| `vector_store.py` | faiss-cpu, numpy | `pipeline.py`, `retriever.py` |
| `retriever.py` | embedder, vector_store | `pipeline.py` |
| `prompt_builder.py` | — (pure logic) | `pipeline.py` |
| `generator.py` | anthropic SDK | `pipeline.py` |
| `pipeline.py` | all above | `app.py` |
| `app.py` | pipeline, streamlit | End user |

---

## 5. Why This Design Is Suitable for the Domain

### Budget PDF (252 pages)
The Ghana Budget Statement is a formal policy document with distinct thematic
sections (Revenue, Expenditure, Debt, Sectors). Paragraph-level chunking
aligns with this structure — each chunk captures a complete policy point,
enabling precise retrieval for questions like "What is the education allocation?".

### Election CSV
Electoral data is tabular and each row is a complete fact (candidate, region,
votes). Converting rows to natural-language sentences allows the same
embedding/retrieval pipeline to work uniformly across both data types. This
avoids a two-system architecture and simplifies maintenance.

### Combined Multi-source RAG
Users can ask cross-domain questions (e.g., "How does the education budget
compare to the percentage of NDC votes in the Northern Region?") because
both data sources are indexed in the same FAISS store with source metadata,
allowing the retriever to pull relevant chunks from either source.

---

## 6. Innovation Component (Part G)

Two innovations are implemented:

### 6.1 Conversation Memory
- Stores up to 5 recent Q&A turns in memory
- Injects history before each new question
- Enables coherent multi-turn conversations without re-asking context

### 6.2 Feedback Loop
- Each response has thumbs up / thumbs down buttons
- Feedback is logged to `logs/feedback.jsonl` with timestamp, question, answer, rating
- Negative feedback entries identify retrieval failures
- This data could be used to fine-tune embeddings or adjust thresholds in production
