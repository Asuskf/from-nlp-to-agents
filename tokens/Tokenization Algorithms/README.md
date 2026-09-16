# Lab · Tokenization Algorithms Compared

← [Back to Track 1 · Tokens](../README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asuskf/from-nlp-to-agents/blob/main/tokens/Tokenization%20Algorithms/Tokenization_Algorithms_Compare.ipynb)

> The tokenizer is the engine no one looks at until something breaks. This lab opens it and
> implements the four paradigms that matter, each with the maths that justifies it.

## Objectives

By the end of this lab you will be able to:

1. **Implement Byte-Level BPE (BBPE).** Build the merge loop from frequency statistics and
   explain the *frequency-maximization* objective behind it — and why operating on bytes
   makes the vocabulary closed under any input, in any language or script.
2. **Implement the Unigram model used by SentencePiece.** Score candidate segmentations
   probabilistically and select the best one with **Viterbi decoding**, rather than greedily
   merging pairs as BPE does.
3. **Reason about unified multimodal tokenization.** Understand how text, image and audio
   are projected into a single discrete vocabulary, and what a *unified discrete
   projection* buys a multimodal model.
4. **Evaluate the token-free (byte-to-byte) paradigm.** Model conditional probability
   directly over bytes, and weigh the removal of the tokenizer against the longer sequences
   it produces.
5. **Choose deliberately.** Given a domain — code, multilingual text, biomedical notes,
   audio — argue which tokenization strategy fits and what it will cost in sequence length.

## Structure of the notebook

| Section | Algorithm | Mathematical foundation |
|---------|-----------|-------------------------|
| 1 | Byte-Level Byte Pair Encoding | Frequency maximization |
| 2 | SentencePiece / Unigram | Viterbi decoding |
| 3 | Unified multimodal tokenization | Unified discrete projection |
| 4 | Token-free, byte-to-byte | Byte-level conditional probability |

Each section pairs a description with a runnable implementation, so the difference between
"BPE merges frequent pairs" and "Unigram picks the most probable segmentation" becomes a
behavioural difference you can observe on the same input string.

## Requirements

`pydantic`, standard library `math`. No external models downloaded.

## Next

→ [GenAI Unit Economics](../GenAI_Unit_Economics_Model.ipynb) — now that you know how many
tokens a string becomes, find out what that costs.
