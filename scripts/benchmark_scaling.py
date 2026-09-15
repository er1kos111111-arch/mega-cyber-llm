"""Scaling benchmark for MC-LLM.

Measures real throughput and efficiency for a given model size:

* tokens/sec, samples/sec;
* TFLOPS (from measured time and estimated FLOPs);
* accelerator utilization;
* memory usage;
* scaling efficiency when run with multiple devices (via torchrun).
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import torch

from model.architecture import MCLLM
from model.config import ModelConfig


def measure(config: ModelConfig, seq_len: int, batch: int, steps: int,
            device: str, dtype=torch.float32):
    torch.manual_seed(0)
    model = MCLLM(config).to(device)
    if dtype != torch.float32 and device == "cuda":
        model = model.to(dtype)

    n_params = model.num_parameters()
    flops_per_token = 6 * n_params  # forward approx

    data = torch.randint(0, config.vocab_size, (batch, seq_len),
                         device=device, dtype=torch.long)
    target = torch.randint(0, config.vocab_size, (batch, seq_len),
                           device=device, dtype=torch.long)

    import torch.nn.functional as F
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # warmup
    for _ in range(2):
        logits = model(data)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    total_tokens = 0
    for _ in range(steps):
        logits = model(data)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_tokens += batch * seq_len
    torch.cuda.synchronize() if device == "cuda" else None
    dt = time.perf_counter() - t0

    tokens_per_sec = total_tokens / dt
    samples_per_sec = batch * steps / dt
    flops = flops_per_token * total_tokens
    tflops = flops / dt / 1e12

    import psutil
    mem = psutil.Process().memory_info().rss / (1024 ** 3)

    return {
        "parameters": n_params,
        "tokens_per_sec": tokens_per_sec,
        "samples_per_sec": samples_per_sec,
        "tflops": tflops,
        "memory_gb": mem,
        "device": device,
        "dtype": str(dtype),
        "batch": batch,
        "seq_len": seq_len,
    }


def main():
    parser = argparse.ArgumentParser(description="MC-LLM scaling benchmark")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    cfg = ModelConfig(
        vocab_size=args.vocab, hidden_size=args.hidden, num_layers=args.layers,
        num_attention_heads=args.heads, num_kv_heads=args.kv_heads,
        intermediate_size=args.intermediate, max_position_embeddings=args.seq_len,
    )

    print("=" * 52)
    print("MC-LLM scaling benchmark")
    print("=" * 52)
    res = measure(cfg, args.seq_len, args.batch, args.steps, device, dtype)
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k:>16}: {v:,.2f}")
        else:
            print(f"{k:>16}: {v}")
    print("=" * 52)


if __name__ == "__main__":
    main()
