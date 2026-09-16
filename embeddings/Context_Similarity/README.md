# Lab 4 · Context Similarity

### Resolving polysemy: context as gravity, dynamic vectors and production pipelines

← [Back to Track 2 · Embeddings](../README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/embeddings/Context_Similarity/Context_Similarity.ipynb)

> Labs 1–3 gave every word one fixed vector. This lab breaks that assumption on purpose,
> then repairs it with the mechanism that made transformers work.

## Objectives

By the end of this lab you will be able to:

1. **Demonstrate the failure of static embeddings.** Show mathematically why one vector per
   word cannot represent a polysemous term, and why the "old world" approach breaks on real
   corpora.
2. **Explain context as gravity.** Describe how surrounding tokens pull a word's
   representation toward one meaning cluster or another.
3. **Compute a dynamic vector with self-attention.** Apply the attention formula to produce
   a context-dependent representation, and watch the same word land in two different
   regions depending on its sentence.
4. **Visualize the gravitational split.** Plot the separation between meaning clusters and
   read the geometry directly.
5. **Design a pipeline for a specialized domain.** Architect a contextual pipeline that
   drives false positives toward zero where a static-vector system would confuse homonyms.
6. **Retrieve over dynamic vectors.** Run semantic search against contextual embeddings —
   the retrieval core of RAG.

## Notebook structure

| Act | Focus |
|-----|-------|
| 1 | **Context as gravity** — the end of the isolated word; the maths of the old world and why it breaks |
| 2 | **The dynamic vector** — self-attention, the "bank" experiment, visualizing the split |
| 3 | **Impact on specialized domains** — pipeline architecture, zero false positives in production |
| 4 | **Retrieving the information** — semantic search over dynamic vectors |

## The architectural takeaway

Retrieval quality is decided before the vector database is ever queried — at the moment you
choose whether your representation is aware of context. Everything track 3 does with HNSW,
reranking and orchestration is built on top of this choice.

## Requirements

`numpy`, `matplotlib`.

## Next

→ [Lab 5 · Simulating RAG Vector Mechanics](../Simulating_RAG_vector_mechanics/) — what
happens when the query lands somewhere the space is empty.
