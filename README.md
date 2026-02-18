# NanoGPT-Surgery 🔬
**From-Scratch GPT Implementation with LoRA Fine-Tuning**

## Project Overview
This repository contains a minimal, from-scratch implementation of a GPT-style language model using PyTorch. The primary goal is to bridge the gap between high-level library usage and low-level architectural understanding by building every component by hand.

Unlike standard tutorials that rely on `AutoModel.from_pretrained`, this project involves writing the model architecture, tokenizer, dataset, and parameter-efficient fine-tuning (LoRA) from the ground up.

## Architecture

### Components

| File | Description |
|---|---|
| `SimpleGPT.py` | A minimal GPT model built on `nn.Module`. Uses token + positional embeddings, a stack of `TransformerEncoderLayer` blocks, layer normalization, and a linear language model head. |
| `CharacterTokenizer.py` | A character-level tokenizer that builds encode/decode lookup tables from the input text. Maps individual characters to integer IDs and back. |
| `GPTDataset.py` | A PyTorch `Dataset` that converts tokenized text into overlapping (input, target) sequence pairs, where the target is the input shifted by one token. Defines a `Tokenizer` protocol for structural subtyping. |
| `LoRALinear.py` | A LoRA (Low-Rank Adaptation) wrapper for `nn.Linear` layers. Freezes the original weights and learns a low-rank delta ($\Delta W = B A$) scaled by $\alpha / r$, enabling parameter-efficient fine-tuning. |
| `main.py` | Entry point that wires everything together: initializes the tokenizer, dataset, and model; injects LoRA into the language model head; runs a training step; and generates text with top-k sampling. |

### How It Works

1. **Tokenization** — `CharacterTokenizer` builds a vocabulary from unique characters in the input text and provides `encode` / `decode` methods.
2. **Dataset** — `GPTDataset` takes the encoded token sequence and creates sliding-window (input, target) pairs of a fixed sequence length.
3. **Model** — `SimpleGPT` is a decoder-style transformer with configurable embedding size (128), attention heads (4), layers (4), and block size (256).
4. **LoRA Injection** — At runtime, `LoRALinear` wraps the model's `lm_head` layer. The original weights are frozen and only the small low-rank matrices A and B are trained.
5. **Training** — Uses `AdamW` with cross-entropy loss, training only the LoRA parameters.
6. **Generation** — Autoregressive text generation with temperature scaling and top-k sampling.

## Requirements
- Python ≥ 3.12
- PyTorch ≥ 2.10
- tiktoken ≥ 0.12

## Quick Start
```bash
cd src
uv sync
uv run main.py
```
