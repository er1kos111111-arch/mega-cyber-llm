"""800B-scale planning & resource estimator for MC-LLM.

Loads the real ``configs/800b.yaml`` (or any config) and computes the
concrete resource requirements.  Critically, it tells you *explicitly*
when your current hardware cannot train the model.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from model.config import RunConfig


def flops_per_token(model) -> float:
    """Approximate FLOPs per token (2 for forward, ~4 for backward → 6N rule)."""
    return 6 * model.total_parameters()


def estimate(model, num_tokens: float, accelerator_tflops: float,
             num_accelerators: int, efficiency: float = 0.40) -> dict:
    n = model.total_parameters()
    bytes_per = {"bf16": 2, "fp16": 2, "fp8": 1, "fp32": 4}
    mem = {k: n * b / 1e9 for k, b in bytes_per.items()}

    optim_states = n * 4 * 3 / 1e9  # fp32 master + m + v (GB)
    gradients = n * 2 / 1e9  # bf16 gradients (GB)

    # total training FLOPs (forward + backward), Chinchilla 6ND
    train_flops = flops_per_token(model) * num_tokens

    peak = accelerator_tflops * num_accelerators * 1e12 * efficiency
    seconds = train_flops / peak if peak > 0 else float("inf")
    days = seconds / 86400

    return {
        "parameters": n,
        "memory": mem,
        "optimizer_memory_gb": optim_states,
        "gradient_memory_gb": gradients,
        "checkpoint_size_gb": n * 2 / 1e9,
        "tokens": num_tokens,
        "train_flops": train_flops,
        "seconds": seconds,
        "days": days,
        "accelerators": num_accelerators,
    }


def main():
    parser = argparse.ArgumentParser(description="800B resource estimator")
    parser.add_argument("--config", default="configs/800b.yaml")
    parser.add_argument("--tokens", type=float, default=15e12,
                        help="total training tokens (default 15T)")
    parser.add_argument("--tflops", type=float, default=989.0,
                        help="sustained BF16 TFLOPS per accelerator (H100 = 989, TPU v4 = 275)")
    parser.add_argument("--num-accel", type=int, default=16384,
                        help="number of accelerators")
    parser.add_argument("--efficiency", type=float, default=0.40)
    args = parser.parse_args()

    run = RunConfig.from_yaml(args.config)
    cfg = run.model

    print("=" * 60)
    print("MEGA-CYBER LLM — target model")
    print("=" * 60)
    print(f"config:           {args.config}")
    print(f"hidden_size:      {cfg.hidden_size}")
    print(f"num_layers:       {cfg.num_layers}")
    print(f"num_heads:        {cfg.num_attention_heads}")
    print(f"num_kv_heads:     {cfg.num_kv_heads}")
    print(f"intermediate:     {cfg.intermediate_size}")
    print(f"vocab_size:       {cfg.vocab_size}")
    print(f"context length:   {cfg.max_position_embeddings}")
    print(f"TOTAL parameters: {cfg.total_parameters():,} "
          f"({cfg.total_parameters()/1e9:.2f}B)")

    est = estimate(cfg, args.tokens, args.tflops, args.num_accel, args.efficiency)

    print("-" * 60)
    print("Memory (weights only):")
    for dtype, gb in est["memory"].items():
        print(f"  {dtype:>6}: {gb:>12.2f} GB")
    print(f"  optimizer (AdamW): {est['optimizer_memory_gb']:>12.2f} GB")
    print(f"  gradients:         {est['gradient_memory_gb']:>12.2f} GB")
    print(f"  checkpoint (bf16): {est['checkpoint_size_gb']:>12.2f} GB")
    print("-" * 60)
    print(f"training tokens:   {est['tokens']:.1e}")
    print(f"train FLOPs:       {est['train_flops']:.3e}")
    print(f"accelerators:      {est['accelerators']}")
    print(f"efficiency:        {args.efficiency}")
    print(f"estimated time:    {est['days']:.1f} days "
          f"({est['days']/365:.1f} years)")

    # honesty check: how many accelerators would be needed for ~60 days?
    target_seconds = 60 * 86400
    needed = (est["train_flops"] / (args.tflops * 1e12 * args.efficiency * target_seconds))
    print("-" * 60)
    print(f"To finish in ~60 days you need ~{needed:,.0f} accelerators")
    print(f"(at {args.tflops} TFLOPS each, {args.efficiency:.0%} efficiency).")
    print("=" * 60)
    print("800B CANNOT be trained on a single machine or a small node.")
    print("Use configs/800b.yaml + the JAX/TPU backend on a large TPU pod")
    print("(TPU Research Cloud) or a multi-thousand-GPU cluster.")
    print("=" * 60)


if __name__ == "__main__":
    main()
