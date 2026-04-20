# Manual Experiment Log
## IT3241 – Introduction to Artificial Intelligence
### RAG System Chatbot — Experiment Records

**Student Name:** Daniel Kingsley Bright Amusah
**Index Number:** 10012300036

> **Note:** These are manually written experiment observations, NOT AI-generated summaries.

---

## Experiment 1 — Chunking Strategy Comparison (Part A)

**Date:** 15 April 2026  
**Objective:** Compare retrieval quality across three chunking strategies.

**Query used:** "What is the government's plan for reducing the fiscal deficit in 2025?"

### Fixed-Size Chunking (512 chars, 64 overlap)
- **Chunks created:** ~4,800 (PDF pages only)
- **Top result similarity:** 0.6821
- **Observation:** Retrieved fragment split mid-sentence: "...the deficit target is 4.7% of GDP which represents a significant improvement from the 7.8% recorded in—" (chunk cut off). The overlap of 64 chars partially recovered the boundary but the result felt incomplete.
- **Verdict:** Acceptable baseline. Loses semantic meaning at boundaries.

### Sentence-Based Chunking (5 sentences, 1 overlap)
- **Chunks created:** ~6,100
- **Top result similarity:** 0.6643
- **Observation:** Results were more complete sentences, but similarity scores slightly lower because each chunk is narrower. Works well for factual one-liners (e.g. election results).
- **Verdict:** Better for precise factual queries; not ideal for budget policy explanations.

### Paragraph-Based Chunking (primary strategy)
- **Chunks created:** ~2,300 (budget PDF only — CSV rows treated as atomic)
- **Top result similarity:** 0.7104
- **Observation:** Retrieved a full, coherent paragraph explaining the deficit reduction roadmap. No mid-sentence cuts. Context felt natural and complete.
- **Verdict:** Best overall. Chosen as default strategy.

**Conclusion:** Paragraph chunking provides the highest similarity scores and most coherent context for the Budget PDF, which is structured around thematic paragraphs. Sentence chunking is useful for the election CSV rows.

---

## Experiment 2 — Query Expansion Impact (Part B)

**Date:** 15 April 2026  
**Objective:** Measure retrieval recall improvement with synonym expansion.

**Baseline query:** "How much did Ghana borrow in 2025?"

| Mode | Top-1 Score | Top-1 Source | Answer relevant? |
|------|-------------|--------------|-----------------|
| No expansion | 0.5932 | Budget p.47 | Partially (indirect) |
| With expansion | 0.7218 | Budget p.53 | Yes (direct debt figures) |

**Expansion added terms:** `debt loan financing borrowing liability`

**Observation:** Without expansion, the retriever found paragraphs mentioning "borrow" but missed sections titled "Financing Requirement" and "Debt Management". With expansion, those sections ranked in top-3.

**Conclusion:** Synonym expansion improved recall for budget-domain terminology where synonyms (debt ↔ borrow) are common.

---

## Experiment 3 — Prompt Template Comparison (Part C)

**Date:** 15 April 2026  
**Objective:** Compare output quality across three prompt templates.

**Query:** "How did NDC perform in the 2020 Ashanti Region election?"

### Template A: Standard
> Response: "According to Chunk 2, in the 2020 Ghana presidential election, John Dramani Mahama of the NDC party received 312,818 votes (26.91%) in the Ashanti Region."
- Clean, direct. No reasoning shown. ✅

### Template B: Chain-of-Thought
> Response: "Let me check the context. Chunk 1 shows Nana Akufo-Addo of NPP received 849,066 votes. Chunk 2 confirms John Dramani Mahama of NDC received 312,818 votes — that is 26.91%. So the NDC performed significantly weaker in Ashanti, a traditional NPP stronghold."
- More insightful; shows reasoning. Takes slightly longer. ✅

### Template C: Strict Factual
> Response: "[Chunk 2] NDC: 312,818 votes (26.91%), Ashanti 2020."
- Extremely brief. Accurate but loses narrative context. ⚠️

**Best template for policy questions:** chain_of_thought  
**Best template for data lookups:** strict_factual  
**Default (balanced):** standard

---

## Experiment 4 — Adversarial Queries (Part E)

**Date:** 15 April 2026  
**Objective:** Test robustness against ambiguous / misleading queries.

### Adversarial Query 1: "Who won the 2025 elections?"
- **RAG response:** "I don't have enough information in the provided documents to answer that. The election dataset only covers years up to 2020 and 2024. There is no 2025 election record."
- **Pure LLM response:** "John Dramani Mahama of the NDC won the 2024 Ghana presidential election." (Correct factually but confused 2025 with 2024.)
- **Analysis:** RAG correctly declined for the exact year 2025. Pure LLM hallucinated a year mapping. RAG = ✅, Pure LLM = ⚠️

### Adversarial Query 2: "What is Ghana's total GDP in dollars?"
- **RAG response:** "The 2025 Budget Statement mentions GDP growth targets and GHS-denominated figures, but does not state total GDP in US dollars. Based on the provided documents, the GDP growth target is 4.0% for 2025."
- **Pure LLM response:** "Ghana's GDP is approximately $77 billion USD (2023)." (Plausible but unverifiable from our docs.)
- **Analysis:** RAG correctly qualified its answer. Pure LLM gave a number without source — could be outdated or wrong. RAG = ✅, Pure LLM = ❌ (hallucination risk)

### Adversarial Query 3: "Compare Ghana's results with Nigeria's"
- **RAG response:** "I don't have enough information in the provided documents to answer that. The dataset only contains Ghana election results and Ghana budget data. No Nigerian data is present."
- **Pure LLM response:** Provided a lengthy comparison mixing real and potentially fabricated statistics.
- **Analysis:** RAG correctly identified the out-of-scope request. RAG = ✅

---

## Experiment 5 — Failure Case Analysis (Part B)

**Date:** 15 April 2026  
**Objective:** Identify and fix retrieval failure cases.

**Failure Case:** Query "education spending" returned three near-identical chunks from the same paragraph, extracted by overlapping fixed-size chunking. All three results were ranked 1-3 but were essentially the same passage.

**Root Cause:** Fixed-size chunking with overlap created duplicate windows over the same passage. FAISS ranked them all highly because they all contained the target phrase.

**Fix Implemented:** `_deduplicate()` in `retriever.py` — computes Jaccard word overlap between results and drops any result that shares > 85% word overlap with an already-selected result. After the fix, top-5 results came from 5 distinct sections of the document.

**Before fix:** Top-3 results all similarity ~0.72, same paragraph text  
**After fix:** Top-3 results from pages 89, 112, 134 — diverse, complementary context

---

## Experiment 6 — Memory (Part G)

**Date:** 15 April 2026  
**Objective:** Test multi-turn conversation coherence.

**Turn 1:** "What was Akufo-Addo's vote count in Ashanti in 2020?"  
**Answer:** "849,066 votes (72.99%)"

**Turn 2 (with memory):** "How does that compare to his national total?"  
**Answer (with memory):** Correctly referenced Ashanti figure from Turn 1 and retrieved national totals — coherent follow-up.

**Turn 2 (without memory):** "How does that compare to his national total?"  
**Answer (no memory):** Lost context, gave unrelated answer about another candidate.

**Conclusion:** Memory-augmented RAG substantially improves multi-turn coherence for analytical follow-up questions.
