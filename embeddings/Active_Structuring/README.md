# Lab 3 · Active Structuring

### From unstructured data to actionable semantic pipelines

← [Back to Track 2 · Embeddings](../README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/embeddings/Active_Structuring/Active_Structuring.ipynb)

> The first lab with a deliverable a downstream system could actually consume: a semantic
> pipeline that turns free text into JSON, using geometry instead of regular expressions.

## Objectives

By the end of this lab you will be able to:

1. **Implement semantic projection with the dot product.** Measure how strongly a document
   activates a target concept inside an embedding space — a scalar that answers "does this
   text talk about X?" without any keyword ever appearing.
2. **Build an active structuring pipeline.** Convert those activations into a structured
   JSON schema using geometric thresholds, producing output a database or an API can accept.
3. **Justify the approach.** Explain why threshold-on-activation generalizes where keyword
   matching and regex break — synonyms, paraphrase, domain jargon, misspellings.

## Notebook structure

| Level | Focus |
|-------|-------|
| 0 | The chaos vs. the schema — what unstructured input really looks like |
| 1 | Semantic projection — the dot product as a concept detector |
| 2 | Active structuring pipeline — activations → JSON schema |

## What you build

A semantic processing pipeline that simulates one of the final stages of an LLM
application: given an unstructured document — a clinical report, a support ticket, a source
file, a business memo — decide which concepts it contains and emit a typed record
describing it.

This is the shape of most real extraction systems in production, minus the learned
embedding model, which arrives in [lab 6](../embeddings_pipeline_clinico/).

## Requirements

`numpy`, `json` (standard library).

## Next

→ [Lab 4 · Context Similarity](../Context_Similarity/) — the failure mode this pipeline
still has: one word, two meanings.
