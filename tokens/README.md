# Track 1 · Tokens — the unit of computation

← [Back to the course](../README.md)

Before a model can *mean* anything, it has to *count* something. This track is about the
atom of every LLM system: the token. What it is, how the text is cut into them, what a
model actually predicts about them, and what each one costs you in production.

## Objectives of this track

By the end you will be able to:

1. **State the objective function of an LLM precisely.** Write the joint probability of a
   sequence as a product of conditional next-token probabilities, and explain why every
   generation step is a distribution over the full vocabulary.
2. **Explain why log-probabilities exist.** Show numerically how joint probability
   collapses toward zero as sequences grow, and why summing logs replaces multiplying
   fractions.
3. **Compare tokenization algorithms on their own terms.** Implement byte-level BPE,
   Unigram/SentencePiece with Viterbi decoding, unified multimodal tokenization and the
   token-free byte paradigm — and articulate the trade-off each one is making.
4. **Connect token counts to money.** Model unit cost per request, separate fixed
   infrastructure from variable token spend, and compute the break-even point of an LLM
   product.

## Why this comes first

Almost every surprising LLM behaviour downstream — a model that "can't count letters", a
prompt that costs three times what you estimated, a multilingual app with terrible latency
— traces back to a tokenization or a probability fact covered here. Track 2 assumes you
already believe that a model manipulates discrete units under a probabilistic objective.

## Contents

| Lab | Objective | Notebook |
|-----|-----------|----------|
| [Chain Rule](Chain%20Rule/) | Derive and implement the chain rule of probability that defines autoregressive generation. | [`LLM_Probability_Chain_Rule.ipynb`](Chain%20Rule/LLM_Probability_Chain_Rule.ipynb) |
| [Tokenization Algorithms](Tokenization%20Algorithms/) | Implement and compare the four dominant tokenization paradigms. | [`Tokenization_Algorithms_Compare.ipynb`](Tokenization%20Algorithms/Tokenization_Algorithms_Compare.ipynb) |
| GenAI Unit Economics | Turn token counts into unit costs, ROI and break-even volume. | [`GenAI_Unit_Economics_Model.ipynb`](GenAI_Unit_Economics_Model.ipynb) |

### GenAI Unit Economics — what this notebook covers

A full viability analysis of a hypothetical LLM application:

- **Microeconomics** — unit cost at the token level.
- **Two consumption patterns** — conversational User↔Agent vs Agent-to-Agent pipelines, which
  have very different input/output token ratios.
- **Cost structure** — fixed infrastructure vs variable token spend, and the resulting ROI.
- **Macroeconomics** — how API price decay and scale change the picture over time.
- **Break-even analysis** — by volume, by price, and by target costing, split per
  consumption pattern.

## Requirements

`numpy`, `pandas`, `matplotlib`, `pydantic`. No API keys, no GPU.

## Next

→ [Track 2 · Embeddings](../embeddings/)
