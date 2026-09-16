# Track 2 · Embeddings — meaning as geometry

← [Back to the course](../README.md)

Track 1 established that a model manipulates discrete tokens under a probabilistic
objective. This track answers the next question: how does a discrete token acquire
*meaning*?

The answer is geometric. Meaning becomes position, similarity becomes angle, and reasoning
becomes linear algebra. Six labs take that claim from a four-dimensional toy space to a
validated pipeline running on a real embedding model.

## Objectives of this track

By the end you will be able to:

1. **Represent meaning as coordinates.** Design latent semantic features by hand, build
   vectors from them, and measure semantic closeness with cosine similarity.
2. **Audit an embedding dimension by dimension.** Inspect what individual coordinates
   encode, and build rule-based classifiers that read tone, urgency and intent straight
   from the numbers.
3. **Use geometry instead of pattern matching.** Replace regex and keyword rules with
   dot-product activations and thresholds, and emit a structured JSON schema from
   unstructured text.
4. **Explain why static vectors fail.** Demonstrate polysemy breaking a fixed-vector model,
   then show how self-attention recomputes a vector from its context at inference time.
5. **Frame hallucination geometrically.** Measure confidence as proximity in latent space,
   identify sparse regions, and show how retrieved context relocates a query — the core
   mechanism behind RAG.
6. **Validate an embedding model mathematically.** Stop guessing which model is better:
   build a golden dataset, compute a separation margin, compare cosine / dot product /
   Euclidean distance, visualize with PCA, and back the pipeline with automated tests.

## Recommended order

The labs build on each other. Take them in this order:

| # | Lab | Objective | Tools |
|---|-----|-----------|-------|
| 1 | [Build Semantic Embeddings](Build_Semantic_Embeddings/) | Vectors, cosine similarity, nearest neighbours and semantic arithmetic in a hand-built space. | NumPy |
| 2 | [From Words to Vector Intelligence](From_Words_to_Vector_Intelligence/) | Reverse-engineer the "semantic DNA" of a word; audit and manipulate dimensions. | NumPy |
| 3 | [Active Structuring](Active_Structuring/) | Unstructured text → semantic projection → structured JSON. | NumPy |
| 4 | [Context Similarity](Context_Similarity/) | Polysemy, self-attention, dynamic vectors, and retrieval over them. | NumPy, Matplotlib |
| 5 | [Simulating RAG Vector Mechanics](Simulating_RAG_vector_mechanics/) | Confidence, empty regions, and semantic grounding by interpolation. | NumPy |
| 6 | [Clinical Embeddings Pipeline](embeddings_pipeline_clinico/) | A real validation pipeline with a real model. *(Spanish)* | sentence-transformers, scikit-learn |

Labs 1–5 use NumPy only, on purpose: with four dimensions you can print every vector and
verify every claim yourself. Lab 6 is the transition to production — the same intuitions,
now tested against an embedding model that was actually trained.

## Requirements

Labs 1–5: `numpy`, `matplotlib`.
Lab 6: adds `sentence-transformers`, `scikit-learn`.

## Next

→ [Track 3 · Final exercise](../final_%20exercise/) — where these vectors get indexed,
retrieved, reranked and served.
