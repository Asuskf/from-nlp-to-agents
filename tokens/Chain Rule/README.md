# Lab · The Chain Rule of Probability in LLMs

← [Back to Track 1 · Tokens](../README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/tokens/Chain%20Rule/LLM_Probability_Chain_Rule.ipynb)

> A language model has one mathematical objective: estimate the joint probability of a
> sequence of tokens. This lab makes that objective concrete enough to compute by hand.

## Objectives

By the end of this lab you will be able to:

1. **Write the factorization.** Express the probability of a sequence
   `W = (w₁ … w_T)` as the product of its conditional next-token probabilities,
   `P(W) = ∏ₜ P(wₜ | w₁ … wₜ₋₁)`.
2. **Locate the model's parameters in the formula.** Identify what the network `θ`
   actually approximates — `P(wₜ | w_<t; θ)` — and why a forward pass returns a
   distribution over the *entire* vocabulary, not a single word.
3. **Simulate autoregressive generation step by step.** Build a small typed `LLMSimulator`
   whose "weights" are hardcoded distributions, then walk a target sequence token by token,
   accumulating the joint probability.
4. **Explain numerical underflow.** Show that joint probability strictly decreases as
   sequences get longer, and justify why production systems add log-probabilities instead
   of multiplying raw probabilities.

## The core idea

The model never predicts the whole text at once. At step `t` it receives the context
`w₁ … wₜ₋₁` and emits a probability distribution over the vocabulary. The probability of
the full sequence is what you get by multiplying the probability it assigned to each token
you actually chose.

In the worked example, `"El gato duerme"` scores `0.5 × 0.6 × 0.8 = 0.24` — a 24% chance of
generating that exact sequence from an empty context.

## Key takeaway

Multiplying fractions shrinks fast. A 200-token answer would produce a number no float can
represent, which is precisely why every real implementation works in log space.

## Requirements

`pydantic`. Runs in seconds on CPU.

## Next

→ [Tokenization Algorithms](../Tokenization%20Algorithms/) — where the `wₜ` in this formula
actually come from.
