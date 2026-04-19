# Academic City RAG Chatbot
##  – Introduction to Artificial Intelligence | End of Semester Exam 2026

**Student Name:** Daniel Kingsley Bright Amusah  
**Index Number:** 10012300036  
**Lecturer:** Godwin N. Danso  

---

## Project Overview

A custom **Retrieval-Augmented Generation (RAG)** chatbot built from scratch for
Academic City University. The system allows users to chat with:

- **Ghana's 2025 Budget Statement and Economic Policy** (252-page PDF)
- **Ghana Presidential Election Results** (CSV: 2020 & 2024 results by region)

> **Key constraint:** No LangChain, LlamaIndex, or pre-built RAG pipelines.
> All components (chunking, embedding, retrieval, prompt construction) are
> manually implemented.

---

## Live Demo

**Deployed App:** [https://ai10012300036-h3ojtgbydesmref7ldxykf.streamlit.app/]  
**GitHub Repo:** [https://github.com/PsalmzyTech/ai_10012300036]

---

## Architecture Summary

```
PDF + CSV  →  pdfplumber/pandas  →  Paragraph Chunker
          →  SentenceTransformer (all-MiniLM-L6-v2, 384-dim)
          →  FAISS IndexFlatIP (cosine similarity)
          →  Query Expansion (synonym injection)
          →  Top-k Retrieval + Deduplication
          →  Prompt Builder (3 template variants)
          →  Claude Haiku LLM (temperature=0.2)
          →  Streamlit UI (chat + evaluation + architecture tabs)
          →  Conversation Memory + Feedback Loop (Part G)
```

See [docs/architecture.md](docs/architecture.md) for the full architecture diagram.

---

## Project Structure

```
.
├── app.py                          # Streamlit UI (main entry point)
├── rag/
│   ├── __init__.py
│   ├── data_loader.py              # Part A: PDF + CSV loading & cleaning
│   ├── chunker.py                  # Part A: 3 chunking strategies
│   ├── embedder.py                 # Part B: sentence-transformers pipeline
│   ├── vector_store.py             # Part B: FAISS vector storage
│   ├── retriever.py                # Part B: top-k + query expansion
│   ├── prompt_builder.py           # Part C: prompt templates
│   ├── generator.py                # LLM (Claude API)
│   └── pipeline.py                 # Part D: full pipeline + Part G memory
├── data/
│   ├── 2025-Budget-Statement-and-Economic-Policy_v4.pdf
│   └── Ghana_Election_Result.csv
├── index/                          # FAISS index (auto-generated)
├── logs/
│   ├── experiment_log.md           # Manual experiment records (Part D)
│   └── feedback.jsonl              # User feedback (Part G)
├── docs/
│   └── architecture.md             # Part F: architecture document
├── .env.example                    # API key template
├── .gitignore
├── requirements.txt
└── README.md
```

---


---

## Setup & Running Locally

### 1. Prerequisites
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com)

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set API key
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 4. Run the app
```bash
streamlit run app.py
```

The first run will build the FAISS index (~2 minutes for 252 PDF pages).
Subsequent runs load the pre-built index instantly.

---

## Key Design Decisions

| Decision | Choice | Justification |
|----------|--------|---------------|
| Embedding model | all-MiniLM-L6-v2 | Local, fast, 384-dim, trained for semantic similarity |
| Vector index | FAISS IndexFlatIP | Exact cosine sim; small corpus → no quantisation needed |
| Chunking | Paragraph (primary) | Budget doc is paragraph-structured; preserves thematic coherence |
| LLM | Claude Haiku 4.5 | Fast, cost-effective, strong instruction-following |
| Temperature | 0.2 | Reduces hallucination in context-grounded mode |
| Retrieval extension | Query expansion | Improves recall for synonym-rich budget terminology |
| Innovation | Memory + feedback | Enables multi-turn chat; feedback loop guides future improvement |

---

## Experiment Summary

See [logs/experiment_log.md](logs/experiment_log.md) for full manual experiment records.

**Key findings:**
1. Paragraph chunking outperforms fixed-size and sentence chunking for the budget PDF
2. Query expansion improves recall by ~22% for budget-domain terminology
3. RAG system correctly declines out-of-scope / hallucination-prone queries
4. Pure LLM hallucinates specific figures not in the training data
5. Conversation memory enables coherent multi-turn Q&A

---

## Video Walkthrough

[Insert 2-minute video link here]

Topics covered:
- Architecture overview
- Live demo of the chatbot
- Query expansion and retrieval transparency
- Adversarial query demonstration
- Evaluation comparison (RAG vs Pure LLM)

---

## Contact

**Email submission to:** godwin.danso@acity.edu.gh  
**GitHub collaborator:** GodwinDansoAcity  
