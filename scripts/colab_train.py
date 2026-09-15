"""End-to-end conversational training driver for Google Colab (and any GPU box).

Phase 1 — build a conversational dataset (no facts, only dialogue):
    generate multi-turn conversations → validate → dedup → shards
    (streaming, resource-monitored, resumable).

Phase 2 — tokenize the conversations with CyberTokenizer.

Phase 3 — pretrain MC-LLM on the conversational corpus on GPU.

Usage:
    python scripts/colab_train.py --conv-tokens 2000000 --steps 500

It auto-detects the GPU / dtype and overrides the model vocab_size to match
the tokenizer.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[colab] GPU: {name} ({vram:.1f} GB VRAM)")
        return "cuda"
    print("[colab] no GPU — falling back to CPU (very slow)")
    return "cpu"


def generate_conversations(args) -> dict:
    from data.conversation.generate_dataset import run, print_report
    from data.resource_monitor import ResourceLimits

    limits = ResourceLimits(
        ram_fraction=args.ram_fraction,
        disk_limit_gb=args.disk_limit_gb,
        max_runtime_seconds=args.max_runtime_minutes * 60,
        max_tokens=args.conv_tokens,
        max_dialogues=0,
    )
    report = run(
        out_dir=args.conv_dir,
        tokenizer_dir=args.tokenizer_dir,
        seed=args.seed,
        vocab_size=args.vocab_size,
        dialogues_per_shard=args.dialogues_per_shard,
        tokens_per_shard=args.tokens_per_shard,
        log_every=args.log_every,
        limits=limits,
    )
    print_report(report)
    return report


def tokenize_conversations(args, tokenizer, vocab_size: int) -> str:
    from data.shard import read_texts, shard_texts

    def iter_conversations():
        for f in sorted(glob.glob(os.path.join(args.conv_dir, "shard_*.jsonl"))):
            yield from read_texts(f)

    os.makedirs(args.data_dir, exist_ok=True)
    print("[colab] tokenizing conversations -> training shards...")
    manifest = shard_texts(iter_conversations(), tokenizer, args.data_dir,
                           vocab_size, max_seq_len=args.seq_len, add_eos=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description="MC-LLM conversational training (Colab)")
    # conversation dataset
    p.add_argument("--conv-tokens", type=int, default=0, help="max conversational tokens (0 = unlimited)")
    p.add_argument("--max-runtime-minutes", type=float, default=0.0)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--dialogues-per-shard", type=int, default=2000)
    p.add_argument("--tokens-per-shard", type=int, default=2_000_000)
    p.add_argument("--ram-fraction", type=float, default=0.85)
    p.add_argument("--disk-limit-gb", type=float, default=6.0)
    p.add_argument("--conv-dir", default="data/conversation/shards")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--tokenizer-dir", default="tokenizer")
    p.add_argument("--data-dir", default="data/shards")
    # model + training
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--intermediate", type=int, default=2048)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    # real-data mode
    p.add_argument("--real-data", action="store_true",
                   help="download real HF data instead of synthetic conversations")
    p.add_argument("--max-rows", type=int, default=0,
                   help="max docs per dataset in real-data mode (0 = all)")
    p.add_argument("--max-files", type=int, default=0,
                   help="max parquet files per dataset in real-data mode (0 = all)")
    args = p.parse_args()

    device = detect_device()
    torch.manual_seed(args.seed)

    if args.real_data:
        # ---- Phase 1+2 (real): download -> tokenizer -> shards ----------
        from scripts.build_real_dataset import download_real_data, train_tokenizer, shard_real_data
        download_real_data(args.raw_dir, args.max_rows, args.max_files)
        vocab_size = train_tokenizer(args.raw_dir, args.tokenizer_dir, args.vocab_size)
        shard_real_data(args.raw_dir, args.tokenizer_dir, args.data_dir,
                        vocab_size, args.seq_len)
        data_note = f"real data (vocab {vocab_size})"
    else:
        # ---- Phase 1: synthetic conversational dataset ------------------
        gen_report = generate_conversations(args)

        # ---- Phase 2: tokenizer + training shards -----------------------
        from tokenizer.tokenizer import CyberTokenizer
        tokenizer = CyberTokenizer(args.tokenizer_dir)
        vocab_size = len(tokenizer)
        print(f"[colab] tokenizer vocab: {vocab_size}")
        tokenize_conversations(args, tokenizer, vocab_size)
        data_note = f"conv tokens: {gen_report['tokens']:,}"

    # ---- Phase 3: build model + train ----------------------------------
    from model.architecture import MCLLM
    from model.config import ModelConfig, TrainingConfig, RunConfig, DistributedConfig

    cfg = ModelConfig(
        vocab_size=vocab_size, hidden_size=args.hidden, num_layers=args.layers,
        num_attention_heads=args.heads, num_kv_heads=args.kv_heads,
        intermediate_size=args.intermediate, max_position_embeddings=args.seq_len,
    )
    training = TrainingConfig(
        micro_batch_size=args.batch, sequence_length=args.seq_len,
        learning_rate=args.lr, max_steps=args.steps,
        warmup_steps=max(10, args.steps // 20),
        log_interval=args.log_interval, save_interval=args.save_interval,
        eval_interval=args.save_interval, dtype="bfloat16",
    )
    dist_cfg = DistributedConfig(use_gradient_checkpointing=args.layers >= 16)
    run = RunConfig(model=cfg, training=training, distributed=dist_cfg,
                    data_dir=args.data_dir, out_dir=args.out)

    from training.train import train
    from data.dataset import build_dataloader
    train_loader = build_dataloader(args.data_dir, args.seq_len, args.batch, seed=args.seed)
    eval_loader = build_dataloader(args.data_dir, args.seq_len, args.batch, seed=args.seed + 1)

    print("=" * 52)
    print("MEGA-CYBER LLM — training")
    print(f"  parameters: {cfg.total_parameters():,}")
    print(f"  vocab: {vocab_size}, context: {args.seq_len}")
    print(f"  {data_note}, steps: {args.steps}")
    print("=" * 52)

    train(run, train_loader=train_loader, eval_loader=eval_loader, resume=False)

    # ---- Phase 4: sample generation ------------------------------------
    from inference.loader import load_model
    from inference.generate import generate
    model, tok, meta = load_model(args.out, args.tokenizer_dir, device=device)
    for prompt in ["Привет! Расскажи о себе.", "Как дела?", "Мне сегодня скучно."]:
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        out = generate(model, ids, max_new_tokens=40, temperature=0.7, seed=1)
        print(f"  Q: {prompt}\n  A: {tok.decode(out[0].tolist())[len(prompt):]}\n")


if __name__ == "__main__":
    main()
