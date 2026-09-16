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

    # build a separate held-out validation set (different seed -> no leakage)
    from data.conversation.generator import DialogueGenerator
    val_gen = DialogueGenerator(seed=args.seed + 999)
    val_dir = os.path.join(os.path.dirname(args.data_dir), "shards_val")
    os.makedirs(val_dir, exist_ok=True)
    print("[colab] building validation shards...")
    def _val_texts():
        for _ in range(2000):
            yield val_gen.generate()["text"]
    shard_texts(_val_texts(), tokenizer, val_dir, vocab_size,
                max_seq_len=args.seq_len, add_eos=True)
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
    p.add_argument("--max-rows", type=int, default=40000,
                   help="max docs per dataset in real-data mode (0 = all)")
    p.add_argument("--max-files", type=int, default=2,
                   help="max parquet files per dataset in real-data mode (0 = all)")
    p.add_argument("--english", action="store_true",
                   help="also download a small English subset (Russian is default)")
    p.add_argument("--tokenizer-max-chars", type=int, default=2_000_000,
                   help="max chars used to train the tokenizer (subsample)")
    # SFT mode
    p.add_argument("--sft", action="store_true",
                   help="run SFT after pretraining (makes it a chat assistant)")
    p.add_argument("--sft-synthetic", type=int, default=20000,
                   help="synthetic Russian dialogues for SFT")
    p.add_argument("--sft-epochs", type=int, default=3)
    p.add_argument("--sft-out", default="checkpoints_sft")
    args = p.parse_args()

    device = detect_device()
    torch.manual_seed(args.seed)

    if args.real_data:
        # ---- Phase 1+2 (real): download -> tokenizer -> shards ----------
        from scripts.build_real_dataset import download_real_data, train_tokenizer, shard_real_data
        download_real_data(args.raw_dir, args.max_rows, args.max_files,
                           include_english=args.english)
        vocab_size = train_tokenizer(args.raw_dir, args.tokenizer_dir,
                                     args.vocab_size, args.tokenizer_max_chars)
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

    # tokenizer is shared by all later phases
    from tokenizer.tokenizer import CyberTokenizer
    tokenizer = CyberTokenizer(args.tokenizer_dir)

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
    import os as _os
    val_dir = _os.path.join(_os.path.dirname(args.data_dir), "shards_val")
    train_loader = build_dataloader(args.data_dir, args.seq_len, args.batch, seed=args.seed)
    if _os.path.exists(_os.path.join(val_dir, "shards_manifest.json")):
        eval_loader = build_dataloader(val_dir, args.seq_len, args.batch, seed=args.seed)
    else:
        eval_loader = build_dataloader(args.data_dir, args.seq_len, args.batch, seed=args.seed + 1)

    print("=" * 52)
    print("MEGA-CYBER LLM — training")
    print(f"  parameters: {cfg.total_parameters():,}")
    print(f"  vocab: {vocab_size}, context: {args.seq_len}")
    print(f"  {data_note}, steps: {args.steps}")
    print("=" * 52)

    train(run, train_loader=train_loader, eval_loader=eval_loader, resume=False)

    # ---- Phase 5: SFT (optional) --------------------------------------
    if args.sft:
        from data.sft_data import build_sft_messages
        from post_training.sft import sft_train
        from inference.loader import load_model as _load_model
        print("\n" + "=" * 52)
        print("SFT: turning the model into a chat assistant")
        print("=" * 52)
        # load the freshly pretrained model from the checkpoint
        model, _, _ = _load_model(args.out, args.tokenizer_dir, device=device)
        messages = build_sft_messages(synthetic_n=args.sft_synthetic, seed=args.seed)
        sft_train(model, tokenizer, messages, out_dir=args.sft_out,
                  epochs=args.sft_epochs, batch_size=args.batch, device=device)

    # ---- Phase 6: chat-style demo -------------------------------------
    from inference.loader import load_model
    from inference.generate import generate
    demo_dir = args.sft_out if args.sft else args.out
    model, tok, meta = load_model(demo_dir, args.tokenizer_dir, device=device)
    demo_msgs = [{"role": "user", "content": "Привет! Расскажи о себе."},
                 {"role": "user", "content": "Как дела?"},
                 {"role": "user", "content": "Мне сегодня скучно, чем заняться?"}]
    print("\n" + "=" * 52)
    print("Chat demo:")
    for m in demo_msgs:
        ids = tok.tokenize_chat([m], add_generation_prompt=True)
        prompt_len = len(ids)
        in_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate(model, in_ids, max_new_tokens=60, temperature=0.7,
                       eos_token_id=tok.eos_token_id, top_p=0.95, seed=1)
        reply = tok.decode(out[0].tolist()[prompt_len:]).strip()
        print(f"  User: {m['content']}\n  Assistant: {reply}\n")


if __name__ == "__main__":
    main()
