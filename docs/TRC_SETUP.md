# Training MC-LLM on Google TPU Research Cloud (TRC)

This guide takes you from "no TPU" to a running MC-LLM training job on the
TPU Research Cloud, using the JAX backend (`training/tpu_backend.py`).

## What TRC gives you

TRC provides **free** TPU pods (v3-8, v4-8, sometimes v4-16/32, and, for
approved scale projects, larger v4/v5p slices). Each pod is a Linux **TPU VM**
with the TPU attached directly (no separate host).

Realistic sizes on standard TRC quotas:

| Pod  | HBM total | Trainable model |
|------|----------:|-----------------|
| v3-8 |   128 GB  | up to ~3B       |
| v4-8 |   256 GB  | up to ~7B       |
| v4-16/32 | 512 GB–1 TB | 7B–70B    |

`800B` is **not** possible on TRC — it needs thousands of chips.

---

## Step 1 — Apply for TRC

1. Go to <https://sites.research.google/trc/> and apply with a short research
   proposal. Approval usually takes days–weeks.
2. Once approved you get a **Google Cloud project** with free TPU quota.
   Note the project id and enable billing (free quota is applied, but billing
   must be linked).

## Step 2 — Create a TPU VM

Install the Google Cloud SDK (`gcloud`) and log in:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

Create a TPU VM (example: v3-8 in us-central2-b):

```bash
gcloud compute tpus tpu-vm create mc-llm-tpu \
  --zone us-central2-b \
  --accelerator-type v3-8 \
  --version tpu-vm-base
```

> `tpu-vm-base` is a plain Debian image; we install JAX ourselves. For v4,
> use `--accelerator-type v4-8` and a `us-central2-b`/`us-central2` zone.

List / delete VMs:

```bash
gcloud compute tpus tpu-vm list --zone us-central2-b
gcloud compute tpus tpu-vm delete mc-llm-tpu --zone us-central2-b
```

## Step 3 — SSH into the VM and get the code

```bash
gcloud compute tpus tpu-vm ssh mc-llm-tpu --zone us-central2-b
```

Upload the project (pick one):

```bash
# (a) git — recommended
git clone https://github.com/YOUR_USER/mega-cyber-llm.git && cd mega-cyber-llm

# (b) scp the local folder
gcloud compute tpus tpu-vm scp --recurse . mc-llm-tpu:~/mega-cyber-llm --zone us-central2-b
```

## Step 4 — One-command bootstrap

Inside the VM:

```bash
bash scripts/run_tpu.sh smoke
```

This installs `jax[tpu]`, prints the TPU devices, and runs a 20-step JAX
smoke test. If you see `jax.devices()` list 8 TPU devices and the loss
decreases, the backend works.

## Step 5 — Tokenizer + data (do this on the VM)

Put your raw corpus under `data/raw/` (`.txt`/`.jsonl`/`.md`/`.html`). Then:

```bash
bash scripts/run_tpu.sh prepare
```

which runs `train_tokenizer.py` (from-scratch CyberTokenizer) and
`prepare_data.py` (clean → filter → dedup → tokenize → shards).

> To reuse a tokenizer already trained locally, upload the `tokenizer/`
> folder and run only `prepare_data.py`.

## Step 6 — Train

```bash
# 100M config (default)
bash scripts/run_tpu.sh train 100m

# or any config directly
python3 -m training.tpu_backend --config configs/1b.yaml --data-dir data/shards \
    --steps 20000 --batch 8 --out checkpoints_tpu
```

Key flags:

| Flag | Meaning |
|------|---------|
| `--config`   | YAML config (`configs/100m.yaml` … `configs/800b.yaml`) |
| `--data-dir` | where the token shards live |
| `--steps`    | number of optimizer steps |
| `--batch`    | sequences per TPU core (× 8 cores = global batch) |
| `--lr`       | peak learning rate |
| `--no-pmap`  | force single-device (debug only) |
| `--smoke`    | random-token smoke test (no data needed) |

## Step 7 — Monitor, resume, checkpoint

* Checkpoints are written to `checkpoints_tpu/` (`step_*.npz`, `final.npz`).
* `tpu_backend.py` uses `jax.pmap` for data-parallel training across all 8
  cores automatically (gradients are `pmean`-averaged).
* To resume, restore `final.npz` via `load_params(path, template)` and
  continue the step counter (the loop is deterministic given `--seed`).

## Step 8 — Keep the VM alive / detach

TRC quota is time-limited — **stop or delete the VM when not training** or
you'll burn your free quota:

```bash
gcloud compute tpus tpu-vm stop mc-llm-tpu --zone us-central2-b
```

Use `tmux`/`screen` inside the VM to detach long runs:

```bash
tmux new -s train
python3 -m training.tpu_backend --config configs/100m.yaml ...
# Ctrl-b d to detach; tmux attach -t train to return
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `jax.devices()` is empty | VM image/zone has no TPU; check `--accelerator-type` |
| `libtpu.so not found` | reinstall: `pip install "jax[tpu]" -f …/libtpu_releases.html` |
| OOM | reduce `--batch`, shorten `--seq-len`, or use a bigger pod |
| slow first step | JAX compiles the graph once; that's normal |
| `corpus too small` | shards have fewer tokens than `seq_len`; regenerate with more data |

## Honest caveat

The JAX backend is a **complete implementation but was not executed on real
TPU hardware at development time** (no JAX/TPU available locally). The first
`run_tpu.sh smoke` run is the validation step — expect minor fixes there.
The PyTorch path (`scripts/train.py`) is fully tested on CPU.
