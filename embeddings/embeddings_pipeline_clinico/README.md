# Lab 6 · Pipeline Clínico de Embeddings

### Algoritmos de similitud y evaluación: de "adivinar el modelo" a *probarlo* matemáticamente

← [Volver al Track 2 · Embeddings](../README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/embeddings/embeddings_pipeline_clinico/lab_embeddings_pipeline_clinico.ipynb)

> Primer laboratorio del track con **modelos reales**. Los cinco anteriores construyeron la
> intuición con NumPy; aquí esa intuición se somete a prueba y se convierte en un criterio
> de decisión defendible.

## El problema que resuelve

**Ceguera de dominio.** Demostrar *cuantitativamente* por qué un modelo de embeddings de
propósito general (`all-MiniLM-L6-v2`) falla al separar jerga clínica del ruido, y por qué
un modelo especializado (`pritamdeka/S-PubMedBert-MS-MARCO`) es la topología geométrica
correcta para un clasificador médico.

## Objetivos

Al terminar este laboratorio serás capaz de:

1. **Implementar las tres métricas de distancia desde cero** — similitud coseno, distancia
   euclídea y producto punto — y argumentar por qué en NLP se usa coseno casi siempre.
2. **Cargar y comparar dos espacios latentes distintos** con `sentence-transformers`,
   entendiendo que la diferencia entre ellos es de *geometría*, no de velocidad.
3. **Construir un Golden Dataset** con *ground truth* y distractores léxicos — textos que
   comparten vocabulario pero no significado, que es donde los modelos generales se rompen.
4. **Auditar qué es realmente un embedding**: forma, dimensionalidad y determinismo del
   vector que devuelve el modelo.
5. **Calcular el Margen de Separación** — la métrica central del lab:

   ```
   Margin = media(cos intra-clase)  −  media(cos inter-clase)
   ```

   generalizada a N documentos. **El modelo con mayor margen es la arquitectura correcta.**
6. **Visualizar la separación con PCA** y confirmar con los ojos lo que la métrica afirma.
7. **Respaldar el pipeline con tests automáticos**, porque una métrica sin tests es una
   opinión con decimales.
8. **Pasar de la validación al producto**: usar los embeddings validados para construir un
   clasificador que prediga clases nuevas.

## Estructura del notebook

| Nivel | Bloque | Qué aprendes |
|-------|--------|--------------|
| 0 | Matemática de la validación | Coseno, euclídea, dot product |
| 1 | Carga de modelos | `sentence-transformers` + Hugging Face |
| 2 | Golden Dataset | Ground truth y distractores léxicos |
| 3 | ¿Qué es un embedding? | Forma, dimensión, determinismo |
| 4 | Métricas de distancia | Las 3 métricas sobre vectores reales |
| 5 | Pipeline de validación | Margen de Separación sobre N documentos |
| 6 | Visualización | Clústeres con PCA |
| 7 | Tests automáticos ✅ | Verificar que el pipeline es correcto |
| 8 | Clasificador real | De los vectores a la predicción |

El notebook cierra con el *takeaway* arquitectónico y ejercicios para extenderlo.

## Requisitos

```bash
pip install sentence-transformers scikit-learn matplotlib numpy
```

La primera ejecución descarga ~500 MB de modelos desde Hugging Face (1–2 min). Pensado para
ejecutarse en orden, celda por celda, preferentemente en Google Colab.

## Siguiente

→ [Track 3 · Ejercicio final](../../final_%20exercise/) — estos vectores, ya validados,
indexados y servidos en un sistema RAG completo.
