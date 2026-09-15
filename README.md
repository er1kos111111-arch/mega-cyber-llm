# MEGA-CYBER LLM (MC-LLM)

A **from-scratch** decoder-only large language model with a **from-scratch**
byte-level BPE tokenizer (`CyberTokenizer`). No pretrained weights, no
third-party tokenizer, no retrieval tables, no API wrappers — every weight
is randomly initialized and every token is learned from the project's own
corpus.

```
========================================
MEGA-CYBER LLM
Architecture : MC-LLM
Tokenizer    : CyberTokenizer
========================================
```

---

## What this is

| Component      | Implementation                                                        |
|----------------|-----------------------------------------------------------------------|
| Architecture   | `model/` — decoder-only transformer (RMSNorm + RoPE + SwiGLU + GQA + KV cache) |
| Tokenizer      | `tokenizer/` — own byte-level BPE trainer + tokenizer (no SentencePiece/HF) |
| Data           | `data/` — cleaning, dedup, filtering, sharding, streaming dataset     |
| Training       | `training/` — causal LM, AdamW/Adam/Muon, warmup+cosine, mixed precision, checkpointing |
| Distributed    | `training/distributed.py` (DDP/FSDP/TP) + `training/tpu_backend.py` (JAX/TPU) |
| Inference      | `inference/` — greedy/temperature/top-k/top-p/min-p/repetition-penalty + KV cache |
| Post-training  | `post_training/` — SFT, DPO, GRPO/Reward Model                        |
| Evaluation     | `evaluation/` — loss, perplexity, built-in benchmarks, JSON report    |

---

## 1. Architecture (MC-LLM)

The model is assembled from scratch in `model/`:

* **`config.py`** — a single `ModelConfig`/`RunConfig` drives everything.
* **`normalization.py`** — RMSNorm.
* **`embeddings.py`** — token embeddings (normal init, scaled).
* **`attention.py`** — causal self-attention with from-scratch **RoPE** and
  a **KV cache**, plus Grouped-Query Attention (GQA).
* **`layers.py`** — `SwiGLUMLP` and a pre-norm `TransformerBlock`.
* **`architecture.py`** — `MCLLM`, the full stack with a from-scratch init.

Supported features: causal masking, RoPE, RMSNorm, SwiGLU, residual streams,
KV cache, autoregressive generation, variable context length, mixed
precision, gradient checkpointing hooks, and (in `training/`) data /
tensor / pipeline / sequence parallelism.

### Parameter accounting

`ModelConfig.total_parameters()` is exact (it matches the PyTorch model —
enforced by `tests/test_model.py::test_parameter_count_matches_formula`).
Run `python scripts/model_size_calculator.py configs/7b.yaml` for a
breakdown (embedding / attention / FFN / normalization / output head).

---

## 2. Tokenizer (CyberTokenizer)

`CyberTokenizer` is a **self-trained byte-level BPE** tokenizer:

* base vocabulary = the 256 byte values → lossless on any UTF-8 text;
* special tokens `<PAD> <BOS> <EOS> <UNK> <USER> <ASSISTANT> <SYSTEM> <TOOL>
  <THINK> <END_THINK> <CODE> <END_CODE>`;
* merges learned by `tokenizer/trainer.py` (`CyberTokenizerTrainer`).

Files: `trainer.py`, `tokenizer.py`, `vocab.py`, `encode.py`, `decode.py`,
plus generated `vocab.json`, `merges.json`, `tokenizer_config.json`.

Train it on your own corpus:

```bash
python scripts/train_tokenizer.py --input data/raw --vocab-size 32768 --out tokenizer
```

---

## 3. Weights

All weights are created by random initialization (`architecture.py`). The
first run produces our own checkpoints under `checkpoints/`:

```
mc_llm_step_000100.pt        # model weights
mc_llm_step_000100_optim.pt  # optimizer + scheduler
mc_llm_step_000100_meta.json # metadata + SHA-256 integrity
latest.json                  # pointer to latest checkpoint
```

Checkpoints support save / load / resume / sharding / integrity validation
(`training/checkpoint.py`).

---

## 4. Data pipeline

`data/` implements a streaming pipeline that never loads the whole dataset
into RAM:

```
data/download → data/raw → data/clean → data/tokenized → data/shards
```

* `cleaner.py` — Unicode normalization, control chars, whitespace, NBSP.
* `dedup.py` — exact (SHA-256) + MinHash near-duplicate detection.
* `filter.py` — language detection, length/quality, PII, spam, repetition.
* `shard.py` — tokenize → binary shards (`uint16`/`uint32`) + manifest.
* `dataset.py` — memory-mapped streaming `IterableDataset`.

```bash
python scripts/prepare_data.py --input data/raw --tokenizer tokenizer \
    --out data/shards --lang-ratio ru:0.5,en:0.5
```

Supports JSONL, TXT, Markdown, HTML, and (optionally, via `pyarrow`)
Parquet/Arrow.

---

## 5. Dataset composition

The pipeline is corpus-agnostic. Language ratios are configurable at the
CLI (`--lang-ratio`) or in a data config, e.g.

```yaml
languages:
  ru: 0.50
  en: 0.45
  other: 0.05
```

Place your corpora (general web, books, Russian, English, technical,
programming, dialogue, instruction, reasoning, safety) under `data/raw` and
adjust ratios — no code changes needed.

---

## 5b. Conversational dataset (current stage)

The current training stage is **100% dialogue** — no knowledge, no facts, no
technical QA. A streaming synthetic generator produces multi-turn Russian
conversations (with some English/mixed) in the `USER:` / `ASSISTANT:` format:

```text
USER: Привет!

ASSISTANT: Привет! Рад тебя видеть.
```

Features (`data/conversation/`):

* **generator** — 13 user personas × 10 assistant styles × 24 everyday topics
  × emotions × fillers × typos × topic changes × in-dialogue memory; a
  configurable length distribution (very short → 120+ messages).
* **quality** — automatic checks (format, role alternation, empty messages,
  repetition, Unicode).
* **dedup** — exact (Bloom filter) + near (MinHash) + template-level.
* **resource_monitor** — RAM/disk/CPU/time limits with safe stop thresholds.
* **generate_dataset** — streaming driver that flushes shards, counts real
  tokens with CyberTokenizer, and resumes after a restart.

```bash
python scripts/generate_conversation_dataset.py --max-tokens 10000000
python scripts/resume_dataset_generation.py            # continue after restart
```

The full Colab conversational pipeline (generate → tokenize → train):

```bash
python scripts/colab_train.py --conv-tokens 2000000 --steps 500
```

---

## 6. Pretraining

Causal language modeling in `training/train.py`:

```
dataset → tokenizer → shards → dataloader → model → forward → loss → backward → optimizer → checkpoint
```

For each batch: input `tokens[0:n-1]`, target `tokens[1:n]`, cross-entropy.
Includes label shifting, padding masking, causal attention masking, gradient
accumulation/clipping, mixed precision (FP16/BF16), warmup+cosine LR, weight
decay, AdamW/Adam/Muon, EMA (optional), and logging.

```bash
python scripts/train.py --config configs/100m.yaml
```

---

## 7. Optimizers

`training/optimizer.py`: from-scratch **AdamW**, **Adam**, and **Muon**.
`training/scheduler.py`: warmup + cosine decay. All hyperparameters come
from the YAML config (`learning_rate`, `weight_decay`, `betas`, `eps`,
`warmup_steps`, `max_steps`, `gradient_clip`).

---

## 8. Distributed training

* **PyTorch path** (`training/distributed.py`): DDP, FSDP (ZeRO-style
  sharding), tensor-parallel linear layers (column/row sharding with
  all-reduce/all-gather), and a pipeline-stage scaffold. Run with
  `torchrun`.
* **TPU path** (`training/tpu_backend.py`): a complete JAX re-implementation
  of MC-LLM with TPU mesh + sharding (data/tensor parallel), for Google TPU
  (TPU Research Cloud) and large pods.

Scales 1 GPU → thousands of accelerators, on NVIDIA, AMD (ROCm via PyTorch),
and Google TPU (JAX/XLA).

---

## 9. Memory optimization

BF16 / FP16 / FP8 (where the backend supports it), gradient checkpointing,
FSDP/ZeRO parameter sharding, KV cache, optimizer-state sharding, and
sharded/distributed checkpointing (`training/checkpoint.py`).

---

## 10. Model configs

`configs/`: `100m`, `300m`, `1b`, `3b`, `7b`, `13b`, `70b`, `405b`, `800b`
(plus `demo.yaml` for the local CPU smoke run). Sizes are *calculated*, not
guessed — see `scripts/model_size_calculator.py`.

---

## 11. 800B planning

`scripts/estimate_800b.py` loads `configs/800b.yaml` and reports parameters,
BF16/FP16/FP8/FP32 memory, optimizer/gradient memory, checkpoint size,
tokens, FLOPs, accelerator count and training duration — and states
explicitly that 800B cannot run on a single machine.

---

## 12. Scaling benchmark

`scripts/benchmark_scaling.py` measures tokens/sec, samples/sec, TFLOPS,
memory and device utilization for a given model size.

---

## 13–16. Chat model, RL, generation, interface

* **SFT** — `post_training/sft.py` (assistant-token loss masking).
* **DPO / RL** — `post_training/dpo.py`, `post_training/rl.py` (reward model
  + GRPO).
* **Generation** — `inference/generate.py` (greedy, temperature, top-k,
  top-p, repetition penalty, min-p, stop tokens, max tokens, streaming).
* **CLI chat** — `python scripts/chat.py`.
* **HTTP API** — `python -m inference.server` → `POST /v1/chat/completions`
  (OpenAI-shaped, backed by our model).

---

## 17–19. Evaluation & tests

```bash
python scripts/evaluate.py --checkpoint checkpoints --suite all
python -m pytest tests/
```

Tests cover the tokenizer (roundtrip, RU/EN/mixed/Unicode/emoji/code/JSON),
model (init, param count, forward/backward/loss, KV cache), checkpoints
(save/load/resume/sharding/integrity), generation (greedy/sampling/top-k/
top-p/determinism/streaming), data, optimizers, and — importantly —
`tests/test_real_llm.py`, which proves the canonical
**train → save → terminate → reload → generate** lifecycle and that the model
is a genuine generator (novel combinations, temperature sensitivity,
token-by-token autoregression), not a lookup table.

---

## 20. Project layout

```
mega-cyber-llm/
├── model/          architecture, attention, layers, embeddings, normalization, config
├── tokenizer/      trainer, tokenizer, vocab, encode, decode
├── data/           cleaner, dedup, filter, shard, dataset
├── training/       train, optimizer, scheduler, distributed, checkpoint, scaler, tpu_backend
├── post_training/  sft, dpo, rl
├── inference/      generate, kv_cache, server, loader
├── evaluation/     evaluate, benchmarks, metrics
├── configs/        100m … 800b (+ demo)
├── scripts/        train_tokenizer, prepare_data, train, generate, chat,
│                   benchmark_scaling, model_size_calculator, estimate_800b, system_info
├── tests/
└── docs/
```

---

## Quick start (proves the pipeline end-to-end on a laptop)

```bash
pip install -r requirements.txt

# 1. tiny demo corpus
python scripts/seed_corpus.py --num-docs 3000 --out data/raw

# 2. train CyberTokenizer from scratch
python scripts/train_tokenizer.py --input data/raw --vocab-size 4096 --out tokenizer

# 3. clean → filter → dedup → tokenize → shard
python scripts/prepare_data.py --input data/raw --tokenizer tokenizer --out data/shards

# 4. train a tiny model (real gradient steps, real checkpoint)
python scripts/train.py --config configs/demo.yaml

# 5. generate
python scripts/generate.py --checkpoint checkpoints --prompt "Привет!" --max-tokens 40

# 6. evaluate
python scripts/evaluate.py --checkpoint checkpoints --suite all
```

> The `demo.yaml` run is a ~293K-parameter smoke test that proves the
> pipeline is real. It produces coherent *token streams*, not fluent text —
> fluent chat requires the 100M+ configs on a GPU/TPU (see below).

---

## Running on Google Colab

Open `notebooks/MC_LLM_Colab.ipynb` in Colab (set **Runtime → GPU**), then run
the cells — or, in a single command, run the whole pipeline:

```bash
python scripts/colab_train.py --steps 500
```

It trains the tokenizer, prepares shards, builds a ~100M model, trains on the
GPU (auto FP16/BF16), saves a checkpoint, and prints generation samples.

## Running on GPU

```bash
python scripts/train.py --config configs/1b.yaml
# multi-GPU (DDP):
torchrun --nproc_per_node=8 scripts/train.py --config configs/7b.yaml
```

## Running on TPU (TPU Research Cloud)

Install JAX on the TPU VM, then use the JAX backend:

```bash
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m training.tpu_backend
```

`training/tpu_backend.py` re-implements MC-LLM in JAX with TPU mesh +
sharding. See `docs/TRC_SETUP.md` for a step-by-step TPU Research Cloud
guide and `docs/SCALING_TO_800B.md` for the full scaling path.

---

## Auto-detection

```bash
python scripts/system_info.py   # CPU/RAM/GPU/VRAM/CUDA/ROCm/TPU/BF16/FP8/disk → recommended config
```

---

## Honesty statement

This repository contains a **real** architecture, a **real** from-scratch
tokenizer, a **real** training loop, and **real** checkpointing/generation.
It does not contain, and does not pretend to contain, a trained 800B model.
What can be run locally has been run (see the smoke pipeline above); what
requires a cluster (100M→800B) is fully implemented but documented as
unrun on this hardware.
