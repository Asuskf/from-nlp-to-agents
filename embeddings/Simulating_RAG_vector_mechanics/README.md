# Lab 5 · Simulating RAG Vector Mechanics

### A conceptual RAG simulation

← [Back to Track 2 · Embeddings](../README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/embeddings/Simulating_RAG_vector_mechanics/Simulating_RAG_vector_mechanics.ipynb)

> A model does not "decide to lie." This lab shows hallucination as what it actually is —
> a geometry problem — and shows retrieval solving it by moving a point.

## Objectives

By the end of this lab you will understand:

- **How an LLM can be viewed geometrically** — as a latent space of concepts rather than a
  database of answers.
- **Why similarity matters when generating text** — proximity to dense, well-populated
  regions correlates with coherent output.
- **How to measure a simple notion of confidence** using cosine similarity, and how to read
  the resulting score.
- **What a sparse region ("empty zone") represents** — a query the space has little
  evidence about, where the model must still predict a next token.
- **How Retrieval-Augmented Generation changes the position of a query** in latent space,
  via vector interpolation toward retrieved evidence.
- **Why RAG adds knowledge without retraining** the model at all.

## Notebook structure

| Level | Focus |
|-------|-------|
| 0 | Measuring geometric confidence — and interpreting the score |
| 1 | Building a tiny latent universe |
| 2 | Simulating an ambiguous query; measuring the nearest concept |
| 3 | Semantic grounding — vector interpolation, applied; what changed |
| — | The RAG pipeline, assembled |

## Scope, stated honestly

The purpose is **not** to eliminate hallucinations. It is to understand *why they happen*,
and why supplying context improves reliability. The latent space here is deliberately
low-dimensional so the mechanism stays visible; real models operate in hundreds or
thousands of dimensions, with the same dynamics.

## Requirements

`numpy`.

## Next

→ [Lab 6 · Clinical Embeddings Pipeline](../embeddings_pipeline_clinico/) — stop simulating
and validate a real model.
