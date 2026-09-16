# Lab 2 · From Words to Vector Intelligence

### Reverse-engineering semantic embeddings like an AI engineer

← [Back to Track 2 · Embeddings](../README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/embeddings/From_Words_to_Vector_Intelligence/From_Words_to_Vector_Intelligence.ipynb)

> Lab 1 treated the vector as a whole. Here you take it apart: what is dimension 3 actually
> storing, and can you classify a message by reading coordinates instead of words?

## Objectives

By the end of this lab you will be able to:

- **Design explicit latent semantic features** that represent concepts numerically.
- **Build custom embedding vectors from scratch** — the "semantic DNA" of a word.
- **Inspect individual dimensions** and state what each coordinate encodes.
- **Create rule-based classifiers from raw coordinates**, detecting tone, urgency and
  intent without reading the text or computing a similarity score.
- **Manipulate latent dimensions** to change semantic context mathematically.
- **Develop transferable intuition** for how learned embeddings behave inside real models.

## Notebook structure

| Level | Focus |
|-------|-------|
| 0 | The anatomy of an embedding |
| 1 | Constructing semantic DNA |
| 2 | Dimensional auditing |
| 3 | Semantic context manipulation |

## Why it matters

Modern models do not process words as text. Every token becomes a high-dimensional vector
whose coordinates carry latent semantic information learned during training. Being able to
reason about — and audit — those internal representations is what separates using an
embedding API from debugging a retrieval system that silently returns the wrong documents.

## Requirements

`numpy`.

## Next

→ [Lab 3 · Active Structuring](../Active_Structuring/) — put those audited dimensions to
work producing structured output.
