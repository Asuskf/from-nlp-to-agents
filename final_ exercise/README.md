# Track 3 · Ejercicio final — RAG de extremo a extremo, capa por capa

### con **HNSW** (hnswlib) · **LangGraph** · **LangSmith** · **OpenRouter**

← [Volver al curso](../README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/final_%20exercise/RAG_HNSW_LangGraph_LangSmith_EJECUTADO_1.ipynb)

> El ejercicio integrador del curso. Las once capas de un sistema RAG, cada una explicada,
> construida y **medida** — incluyendo las que casi nadie enseña: observabilidad, evaluación
> reproducible y análisis de sensibilidad.

## Objetivos

Al terminar este ejercicio serás capaz de:

1. **Diseñar la capa de ingesta** — limpiar y normalizar un corpus, y entender por qué esta
   capa decide más sobre la calidad final de lo que parece.
2. **Elegir una estrategia de chunking con criterio** — comparar tres enfoques, entender el
   compromiso fundamental (contexto vs precisión) y aplicar *contextual chunk headers*, el
   truco que más rinde por línea de código.
3. **Instrumentar la capa de embeddings** — la matemática mínima necesaria, la diferencia
   entre bi-encoder y cross-encoder, y un backend con *fallback* offline.
4. **Construir un índice HNSW** — entender el problema que resuelve, su idea en tres pasos,
   sus tres parámetros (`M`, `ef_construction`, `ef_search`) y visualizar la estructura
   antes de construirla con `hnswlib`.
5. **Recuperar de forma honesta** — trazar la curva `ef_search` ↔ *recall* ↔ latencia,
   validar si HNSW "miente" sobre *tu* corpus, filtrar por metadatos **durante** el
   recorrido (no después) y aplicar MMR cuando el top-k se llena de clones.
6. **Combinar denso y léxico** — Reciprocal Rank Fusion (RRF) sobre BM25 + vectores, y
   reranking con cross-encoder, sabiendo en qué falla cada uno por separado.
7. **Escribir un buen prompt RAG** y servir la generación vía OpenRouter.
8. **Orquestar con LangGraph** — estado, nodos y aristas; implementar **RAG correctivo
   (CRAG)**; y usar *streaming* de estados para ver el grafo pensar.
9. **Observar el sistema con LangSmith** — qué es una traza, por qué importan los
   `run_type`, cómo usar metadatos y etiquetas para comparar experimentos, y cómo
   instrumentar bien un *retriever*.
10. **Evaluar por capas** — métricas de recuperación por separado, *LLM-as-judge* para la
    generación, evaluación reproducible con `langsmith.evaluate`, y un análisis de
    sensibilidad que cuantifica cuánto importa cada decisión de diseño.
11. **Reconocer los siete errores más comunes** en un RAG antes de cometerlos en producción.

## Mapa del cuaderno

| Capa | Contenido |
|------|-----------|
| 0 | Preparación del entorno · dependencias · claves · el decorador `@traceable` |
| 1 | **Ingesta** — corpus, limpieza y normalización |
| 2 | **Chunking** — el compromiso fundamental, tres estrategias, *contextual chunk headers* |
| 3 | **Embeddings** — matemática mínima, bi-encoder vs cross-encoder, fallback offline |
| 4 | **Indexación HNSW** — la idea, los tres parámetros, visualización, construcción |
| 5 | **Recuperación** — curva de recall/latencia, validación honesta, filtrado, MMR |
| 6 | **Híbrida y reranking** — RRF, cross-encoder |
| 7 | **Generación** — OpenRouter, anatomía de un prompt RAG |
| 8 | **Orquestación** — LangGraph, RAG correctivo (CRAG), streaming de estados |
| 9 | **Observabilidad** — LangSmith, trazas, `run_type`, metadatos |
| 10 | **Evaluación** — métricas por capa, LLM-as-judge, `langsmith.evaluate`, sensibilidad |
| 11 | **Cierre** — resumen ejecutable, chuleta de HNSW, siete errores comunes, referencias |

## Requisitos

```bash
pip install hnswlib rank-bm25 sentence-transformers scikit-learn langgraph langchain-core langsmith openai matplotlib numpy
```

### Claves de API (opcionales)

| Variable | Para qué | ¿Obligatoria? |
|----------|----------|---------------|
| `OPENROUTER_API_KEY` | Capa de generación | No — hay *fallback* offline |
| `LANGSMITH_API_KEY` | Trazas y evaluación | No — sin ella se omite el tracing |

El cuaderno está pensado para ejecutarse completo sin claves; con ellas verás además las
trazas y las evaluaciones reales.

## Prerrequisitos del curso

Este ejercicio asume los dos tracks anteriores:
[Tokens](../tokens/) (por qué el chunking y el coste se miden en tokens) y
[Embeddings](../embeddings/) (por qué la recuperación es una operación geométrica).
