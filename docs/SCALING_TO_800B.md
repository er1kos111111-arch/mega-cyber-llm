# Scaling MC-LLM from 100M to 800B

This document describes how to move MC-LLM through the full size ladder and
what changes at each step. The architecture, tokenizer, data pipeline, and
training loop are identical across scales — only the config changes.

## The scaling ladder

| Config | Parameters | Hidden | Layers | Heads | KV heads | Intermediate | Context | Vocab |
|--------|-----------:|-------:|-------:|------:|---------:|-------------:|--------:|------:|
| 100m   |     96.5M  |   512  |   16   |   8   |    4     |     2048     |  2048   | 32k   |
| 300m   |    369M    |  1024  |   24   |  16   |    8     |     3072     |  4096   | 32k   |
| 1b     |    931M    |  1536  |   24   |  24   |    8     |     6144     |  8192   | 32k   |
| 3b     |    3.21B   |  2560  |   32   |  32   |    8     |    10240     |  8192   | 32k   |
| 7b     |    7.25B   |  4096  |   32   |  32   |    8     |    14336     |  8192   | 32k   |
| 13b    |   11.3B    |  5120  |   40   |  40   |    8     |    13824     |  8192   | 32k   |
| 70b    |   69.0B    |  8192  |   80   |  64   |    8     |    28672     | 32768   | 32k   |
| 405b   |  405.8B    | 16384  |  126   | 128   |    8     |    53248     | 131072  | 128k  |
| 800b   |  812.6B    | 20480  |  170   | 128   |   16     |    61440     | 131072  | 256k  |

Verify any config with:

```bash
python scripts/model_size_calculator.py configs/7b.yaml
python scripts/estimate_800b.py
```

## Stage 0 — 100M (single GPU / small TPU slice)

The goal is correctness, not speed. One A100/H100 or a TPU v3-8 fits easily.

* Train the tokenizer on the full corpus (target 32k vocab).
* Prepare shards.
* `python scripts/train.py --config configs/100m.yaml`

Iterate here on loss curves, LR schedule, and checkpoint/resume before
scaling. A 100M model can also run (slowly) on CPU for pure validation.

## Stage 1 — 1B → 7B (data + FSDP parallelism)

Beyond ~1B, one accelerator no longer holds weights + optimizer + gradients,
so you switch on **FSDP** (ZeRO-style sharding) and **gradient checkpointing**:

```bash
torchrun --nproc_per_node=8 scripts/train.py --config configs/7b.yaml
```

* `distributed.fsdp: true` shards parameters/optimizer/gradients across ranks.
* `use_gradient_checkpointing: true` trades compute for activation memory.
* Sequence length grows to 4096/8192; the `data/shard.py` pipeline already
  produces sequences of any length.

## Stage 2 — 70B → 405B (tensor + pipeline parallelism)

At this scale you combine all three dimensions:

* **Tensor parallel** splits each `nn.Linear` across ranks
  (`training/distributed.py::ColumnParallelLinear` / `RowParallelLinear`).
* **Pipeline parallel** splits layers across ranks; config
  `distributed.pipeline_parallel` describes the number of stages.
* **Sequence parallel** shards activations along the sequence dimension.

```bash
torchrun --nproc_per_node=64 scripts/train.py --config configs/70b.yaml
```

## Stage 3 — 800B (TPU pod / multi-thousand-GPU)

Use the JAX/TPU backend (`training/tpu_backend.py`) on a large TPU pod
(TPU Research Cloud) or a multi-thousand-GPU cluster.

The `configs/800b.yaml` `distributed` section describes:

```yaml
distributed:
  tensor_parallel: 16
  pipeline_parallel: 32
  data_parallel: 512
  sequence_parallel: true
```

### Resource math (honest numbers)

From `python scripts/estimate_800b.py`:

* **812.6B parameters**
* BF16 weights: ~1,625 GB; optimizer (AdamW, fp32 master + m + v): ~9,752 GB;
  gradients: ~1,625 GB; checkpoint: ~1,625 GB.
* Training 15T tokens ≈ 7.3×10²⁵ FLOPs.
* On 16,384 H100-class accelerators at ~40% efficiency: ~130 days.
* ~35k accelerators finish in ~60 days.

**This cannot run on a single machine or a small node.** The number of
accelerators, not the code, is the scaling bottleneck — and the code
(`tpu_backend.py` + `distributed.py`) is already written to use them.

## What to change at each stage

| Concern          | 100M→7B            | 70B→800B                      |
|------------------|--------------------|-------------------------------|
| Parallelism      | DDP → FSDP         | + tensor + pipeline + sequence |
| Precision         | BF16               | BF16 (FP8 where supported)     |
| Checkpointing     | per-step file      | sharded + distributed          |
| Data              | 32k vocab          | 128k→256k vocab                |
| Context           | 2k→8k              | 32k→131k (RoPE `rope_theta` raised) |
| LR / batch        | 3e-4 / 0.5–4M      | 1e-4 / 16–33M tokens           |

## Checklist before each scale-up

1. `model_size_calculator.py` reports the expected parameter count.
2. Tokenizer roundtrip tests pass on the new vocab size.
3. Data shards regenerate correctly for the new sequence length.
4. A 1-step train + save + load + generate smoke test passes at the new size.
5. `benchmark_scaling.py` shows throughput scales with added accelerators.
