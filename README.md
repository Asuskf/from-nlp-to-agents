# From NLP to Agents

**A practical, notebook-driven journey from raw tokens to a production-shaped RAG system.**

This repository is a hands-on course. Each notebook is a self-contained lab that builds one
layer of a modern language-model application — and every layer is implemented from first
principles before any library is allowed to hide the mechanics.

You start by counting tokens and multiplying conditional probabilities. You end with an
end-to-end RAG pipeline running on HNSW, orchestrated with LangGraph, traced with LangSmith
and evaluated with an LLM-as-judge.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Who this is for

AI engineers, ML engineers and data scientists who can already write Python and want to
understand *why* embeddings, vector search and RAG behave the way they do — not just how to
call an SDK.

**Prerequisites:** Python, basic NumPy, and comfort with vectors and dot products.
No prior NLP background required.

---

## The learning path

The three tracks are meant to be taken in order. Each one answers a question the previous
one leaves open.

| # | Track | Question it answers | Folder |
|---|-------|---------------------|--------|
| 1 | **Tokens** | How does a model turn text into units it can assign probability to — and what does each unit cost? | [`tokens/`](tokens/) |
| 2 | **Embeddings** | How does meaning become geometry, and how do you validate that geometry mathematically? | [`embeddings/`](embeddings/) |
| 3 | **Final exercise** | How do all those layers compose into a retrieval system you can observe, evaluate and defend? | [`final_ exercise/`](final_%20exercise/) |

Every folder has its own `README.md` stating the objectives of that lesson. Start there.

---

### 1 · Tokens — the unit of computation

| Lab | What you learn | Open |
|-----|----------------|------|
| [Chain Rule](tokens/Chain%20Rule/) | The chain rule of probability — why factorizing a sequence into conditional next-token probabilities *is* the training objective, and why real models sum log-probs instead of multiplying. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/tokens/Chain%20Rule/LLM_Probability_Chain_Rule.ipynb) |
| [Tokenization Algorithms](tokens/Tokenization%20Algorithms/) | BBPE, SentencePiece/Unigram with Viterbi decoding, unified multimodal tokenization, and the token-free byte-to-byte paradigm — implemented, not described. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/tokens/Tokenization%20Algorithms/Tokenization_Algorithms_Compare.ipynb) |
| [GenAI Unit Economics](tokens/) | Token-level unit costs, User↔Agent vs Agent-to-Agent consumption patterns, fixed vs variable cost structure, ROI and break-even analysis. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/tokens/GenAI_Unit_Economics_Model.ipynb) |

### 2 · Embeddings — meaning as geometry

Recommended order:

| # | Lab | What you learn | Open |
|---|-----|----------------|------|
| 1 | [Build Semantic Embeddings](embeddings/Build_Semantic_Embeddings/) | Hand-build a four-dimensional semantic space; measure closeness with cosine similarity; do arithmetic on meaning. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/embeddings/Build_Semantic_Embeddings/Build_Semantic_Embeddings.ipynb) |
| 2 | [From Words to Vector Intelligence](embeddings/From_Words_to_Vector_Intelligence/) | Reverse-engineer what each dimension *means*; audit coordinates; classify tone and intent without reading the text. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/embeddings/From_Words_to_Vector_Intelligence/From_Words_to_Vector_Intelligence.ipynb) |
| 3 | [Active Structuring](embeddings/Active_Structuring/) | Turn unstructured text into a structured JSON schema using dot-product activations and geometric thresholds instead of regex. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/embeddings/Active_Structuring/Active_Structuring.ipynb) |
| 4 | [Context Similarity](embeddings/Context_Similarity/) | Why static vectors break on polysemy; how self-attention makes a vector dynamic; semantic search over contextual embeddings. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/embeddings/Context_Similarity/Context_Similarity.ipynb) |
| 5 | [Simulating RAG Vector Mechanics](embeddings/Simulating_RAG_vector_mechanics/) | Hallucination as a geometry problem: sparse regions, confidence via cosine, and how grounding moves a query inside latent space. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/embeddings/Simulating_RAG_vector_mechanics/Simulating_RAG_vector_mechanics.ipynb) |
| 6 | [Clinical Embeddings Pipeline](embeddings/embeddings_pipeline_clinico/) | **First lab with real models.** Golden dataset, separation-margin metric, cosine vs dot vs Euclidean, PCA, automated tests, and a working classifier. *(Spanish)* | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/embeddings/embeddings_pipeline_clinico/lab_embeddings_pipeline_clinico.ipynb) |

Labs 1–5 run on NumPy alone, deliberately: the point is that you can see every number.
Lab 6 is where `sentence-transformers` enters and the intuitions get tested against a real
embedding model.

### 3 · Final exercise — the whole system

| Lab | What you learn | Open |
|-----|----------------|------|
| [End-to-end RAG](final_%20exercise/) | Eleven layers — ingestion, chunking, embeddings, HNSW indexing, retrieval, hybrid search and reranking, generation, LangGraph orchestration (corrective RAG), LangSmith observability, evaluation, and a design-sensitivity analysis. *(Spanish)* | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/final_%20exercise/RAG_HNSW_LangGraph_LangSmith_EJECUTADO_1.ipynb) |

---

## Repository structure

```
from-nlp-to-agents/
├── tokens/                                 # Track 1 — the unit of computation
│   ├── Chain Rule/                         # Joint probability, factorized
│   ├── Tokenization Algorithms/            # BBPE · Unigram · multimodal · byte-level
│   └── GenAI_Unit_Economics_Model.ipynb    # What a token costs, and when it pays off
│
├── embeddings/                             # Track 2 — meaning as geometry
│   ├── Build_Semantic_Embeddings/          # Vectors, cosine, semantic arithmetic
│   ├── From_Words_to_Vector_Intelligence/  # Dimensional auditing
│   ├── Active_Structuring/                 # Unstructured text → JSON schema
│   ├── Context_Similarity/                 # Polysemy, self-attention, dynamic vectors
│   ├── Simulating_RAG_vector_mechanics/    # Why grounding reduces hallucination
│   └── embeddings_pipeline_clinico/        # Real models + validation pipeline
│
└── final_ exercise/                        # Track 3 — the full RAG system
    └── RAG_HNSW_LangGraph_LangSmith_EJECUTADO_1.ipynb
```

---

## How to run

### Option A — Google Colab (recommended)

Click any **Open in Colab** badge above. Nothing to install; the notebooks that need
dependencies install them in their first cell.

### Option B — Locally

```bash
git clone https://github.com/Asuskf/from-nlp-to-agents.git
```

```bash
python -m venv .venv && source .venv/bin/activate && pip install jupyterlab numpy matplotlib pandas pydantic && jupyter lab
```

On Windows, activate with `.venv\Scripts\activate` instead.

That covers track 1 and track 2 up to lab 5. For the labs that use real models, add:

```bash
pip install sentence-transformers scikit-learn
```

And for the final exercise:

```bash
pip install hnswlib rank-bm25 langgraph langchain-core langsmith openai
```

### API keys

Only the final exercise talks to external services, and both keys are optional — it falls
back to a local backend when they are absent:

| Variable | Used for | Required? |
|----------|----------|-----------|
| `OPENROUTER_API_KEY` | Generation layer | No — offline fallback available |
| `LANGSMITH_API_KEY` | Tracing and evaluation | No — tracing is skipped without it |

Set them as environment variables or Colab secrets. Never commit them; `.env` is already
gitignored.

---

## Conventions used across the labs

- **First principles first.** A concept is implemented in NumPy before any library that
  does the same thing is introduced.
- **Levels, not chapters.** Most labs are structured as `Level 0 → Level N`, each level
  adding one idea and one failure mode.
- **Every claim is measured.** Where a lab says a technique helps, there is a number in the
  notebook showing by how much.

---

## License

MIT — see [LICENSE](LICENSE).
