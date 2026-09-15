"""Parameter count + memory calculator for MC-LLM configurations.

Loads any ``configs/*.yaml`` and prints a breakdown of parameters and
estimated memory, so config values are *calculated* rather than guessed.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import sys

from model.config import RunConfig


def estimate_memory(num_params: int, num_trainable: int, dtype_bytes: int,
                    num_optim_states: int, num_layers: int, seq_len: int,
                    batch: int, hidden: int) -> dict:
    params_gb = num_params * dtype_bytes / 1e9
    grad_gb = num_trainable * dtype_bytes / 1e9
    # optimizer: fp32 master weights + m + v (+ muon buffer) -> 3 states typical
    optim_gb = num_trainable * 4 * num_optim_states / 1e9
    # activations (rough): 2 * batch * seq * hidden * num_layers * dtype
    act_gb = 2 * batch * seq_len * hidden * num_layers * dtype_bytes / 1e9
    return {
        "params": params_gb,
        "gradients": grad_gb,
        "optimizer": optim_gb,
        "activations": act_gb,
        "total": params_gb + grad_gb + optim_gb + act_gb,
    }


def main():
    parser = argparse.ArgumentParser(description="MC-LLM model size calculator")
    parser.add_argument("config", help="path to a config yaml")
    parser.add_argument("--dtype", default="bf16",
                        choices=["fp32", "fp16", "bf16", "fp8"])
    args = parser.parse_args()

    run = RunConfig.from_yaml(args.config)
    cfg = run.model

    bytes_per = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}[args.dtype]

    emb = cfg.embedding_parameters()
    attn = cfg.attention_parameters() * cfg.num_layers
    ffn = cfg.ffn_parameters() * cfg.num_layers
    norm = cfg.normalization_parameters()
    head = cfg.output_head_parameters()
    total = cfg.total_parameters()

    print("=" * 52)
    print("MC-LLM parameter breakdown")
    print("=" * 52)
    print(f"config:            {args.config}")
    print(f"hidden_size:       {cfg.hidden_size}")
    print(f"num_layers:        {cfg.num_layers}")
    print(f"num_heads:         {cfg.num_attention_heads}")
    print(f"num_kv_heads:      {cfg.num_kv_heads}")
    print(f"intermediate:      {cfg.intermediate_size}")
    print(f"vocab_size:        {cfg.vocab_size}")
    print(f"context:           {cfg.max_position_embeddings}")
    print("-" * 52)
    print(f"embedding params:  {emb:>20,}")
    print(f"attention params:  {attn:>20,}")
    print(f"FFN params:        {ffn:>20,}")
    print(f"normalization:     {norm:>20,}")
    print(f"output head:       {head:>20,}")
    print("-" * 52)
    print(f"TOTAL parameters:  {total:>20,}  ({total/1e9:.3f}B)")
    print("=" * 52)

    mem = estimate_memory(total, total, bytes_per, num_optim_states=3,
                          num_layers=cfg.num_layers,
                          seq_len=cfg.max_position_embeddings,
                          batch=run.training.micro_batch_size,
                          hidden=cfg.hidden_size)
    print(f"dtype:             {args.dtype}")
    print(f"model weights:     {mem['params']:.2f} GB")
    print(f"gradients:         {mem['gradients']:.2f} GB")
    print(f"optimizer states:  {mem['optimizer']:.2f} GB")
    print(f"activations(est):  {mem['activations']:.2f} GB")
    print(f"total(est):        {mem['total']:.2f} GB")
    print("=" * 52)


if __name__ == "__main__":
    main()
