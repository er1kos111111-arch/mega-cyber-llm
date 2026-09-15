"""SFT CLI: fine-tune a pretrained MC-LLM into a conversational assistant.

    python scripts/sft_train.py --checkpoint checkpoints --synthetic 20000
    python scripts/sft_train.py --checkpoint checkpoints --synthetic 20000 --persona-max-rows 50000

The result is a chat model that reliably follows the
``<USER> / <ASSISTANT>`` format and answers like a helpful assistant.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch


def main():
    p = argparse.ArgumentParser(description="SFT for MC-LLM")
    p.add_argument("--checkpoint", default="checkpoints",
                   help="directory with the pretrained model")
    p.add_argument("--tokenizer", default="tokenizer/tokenizer_config.json")
    p.add_argument("--out", default="checkpoints_sft")
    p.add_argument("--synthetic", type=int, default=20000,
                   help="number of synthetic Russian dialogues for SFT")
    p.add_argument("--persona-max-rows", type=int, default=0,
                   help="download real PersonaChat dialogues (0 = skip)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SFT] device: {device}")

    from inference.loader import load_model
    model, tokenizer, meta = load_model(args.checkpoint, args.tokenizer, device=device)
    print(f"[SFT] pretrained model loaded: {sum(x.numel() for x in model.parameters()):,} params")

    from data.sft_data import build_sft_messages
    messages = build_sft_messages(synthetic_n=args.synthetic,
                                  persona_max_rows=args.persona_max_rows,
                                  seed=args.seed)
    print(f"[SFT] total examples: {len(messages):,}")

    from post_training.sft import sft_train
    summary = sft_train(
        model, tokenizer, messages,
        out_dir=args.out, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, max_length=args.max_length, seed=args.seed, device=device,
    )

    print("=" * 52)
    print("SFT complete")
    print(f"  validation loss: {summary['sft_loss']:.4f}")
    print(f"  perplexity:      {summary['sft_perplexity']}")
    print(f"  checkpoint:      {summary['checkpoint']}")
    print("=" * 52)
    print(f"\nChat with the SFT model:")
    print(f"  python scripts/chat.py --checkpoint {args.out}")


if __name__ == "__main__":
    main()
