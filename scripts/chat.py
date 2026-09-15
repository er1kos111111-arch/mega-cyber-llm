"""Interactive chat CLI for MC-LLM (SFT model).

    python scripts/chat.py --checkpoint checkpoints_sft

Provides a simple REPL using the MC-LLM chat template.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch

from inference.generate import generate
from inference.loader import load_model


def main():
    parser = argparse.ArgumentParser(description="MC-LLM chat")
    parser.add_argument("--checkpoint", default="checkpoints_sft")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer_config.json")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--system", default="Ты — MEGA-CYBER LLM, дружелюбный и полезный собеседник.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, meta = load_model(args.checkpoint, args.tokenizer, device=device)
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 52)
    print("MEGA-CYBER LLM (SFT)")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Tokenizer:   CyberTokenizer")
    print(f"Parameters:  {n_params:,}")
    print(f"Context:     {model.config.max_position_embeddings}")
    print("=" * 52)
    print("Type 'exit' to quit.\n")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    while True:
        try:
            user = input("User: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.strip().lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user})
        ids = tokenizer.tokenize_chat(messages, add_generation_prompt=True)
        prompt_len = len(ids)
        prompt_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate(model, prompt_ids, max_new_tokens=args.max_tokens,
                       eos_token_id=tokenizer.eos_token_id,
                       temperature=args.temperature, top_p=args.top_p)
        new_ids = out[0].tolist()[prompt_len:]
        reply = tokenizer.decode(new_ids).strip()
        messages.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
