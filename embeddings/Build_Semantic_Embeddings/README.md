# Lab 1 · Build Semantic Embeddings

### Understanding the geometry behind LLM embeddings

← [Back to Track 2 · Embeddings](../README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/embeddings/Build_Semantic_Embeddings/Build_Semantic_Embeddings.ipynb)

> The entry point to the whole track. You design a semantic space by hand, small enough
> that every number is visible, and discover that "understanding meaning" is a geometry
> operation.

## Objectives

By the end of this lab you will be able to:

1. **Represent meaning as a vector.** Define a handful of explicit semantic dimensions and
   place words inside that space by assigning coordinates.
2. **Measure semantic similarity.** Implement cosine similarity and explain why the *angle*
   between two vectors captures meaning better than the distance between them.
3. **Find the nearest semantic neighbour.** Given a query vector, rank the vocabulary and
   retrieve the closest concept — the primitive that every vector database is built on.
4. **Do arithmetic on meaning.** Add and subtract vectors to move through the space, and
   see that analogies become linear operations.
5. **Generalize the intuition.** Explain why the same four principles hold for models with
   768 or 3072 learned dimensions.

## Notebook structure

| Step | What you do |
|------|-------------|
| 1 | Represent meaning as vectors |
| 2 | Measure semantic similarity |
| 3 | Find the closest semantic neighbour |
| 4 | Semantic vector arithmetic |
| — | Why this matters for modern LLMs |

## The takeaway

Our space has four dimensions, chosen by hand. Word2Vec, FastText, OpenAI and Gemini
embeddings learn hundreds or thousands of them automatically. The complexity changes; the
principles do not:

- Words become vectors.
- Similar meanings occupy nearby regions.
- Cosine similarity measures semantic closeness.
- Linear algebra enables semantic reasoning.

That is the foundation under semantic search, recommendation, clustering, vector databases
and every RAG retrieval pipeline.

## Requirements

`numpy`.

## Next

→ [Lab 2 · From Words to Vector Intelligence](../From_Words_to_Vector_Intelligence/) — open
the vector up and find out what each dimension is doing.
