"""Interactive chat CLI for MC-LLM.

    python scripts/chat.py --checkpoint checkpoints

Provides a simple REPL in the MC-LLM chat format.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch

from inference.generate import generate
from inference.loader import load_model


def build_prompt(tokenizer, system: str, history: list) -> str:
    parts = []
    if system:
        parts.append(f"<SYSTEM>\n{system}\n")
    for user_msg, asst_msg in history:
        parts.append(f"<USER>\n{user_msg}\n")
        if asst_msg:
            parts.append(f"<ASSISTANT>\n{asst_msg}\n")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description="MC-LLM chat")
    parser.add_argument("--checkpoint", default="checkpoints")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer_config.json")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--system", default="You are MEGA-CYBER LLM, a helpful assistant.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, meta = load_model(args.checkpoint, args.tokenizer, device=device)
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 52)
    print("MEGA-CYBER LLM")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Tokenizer:   CyberTokenizer")
    print(f"Parameters:  {n_params:,}")
    print(f"Context:     {model.config.max_position_embeddings}")
    print("=" * 52)
    print("Type 'exit' to quit.\n")

    history = []
    while True:
        try:
            user = input("User: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.strip().lower() in ("exit", "quit"):
            break

        prompt = build_prompt(tokenizer, args.system, history) + f"<USER>\n{user}\n<ASSISTANT>\n"
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = generate(model, ids, max_new_tokens=args.max_tokens,
                       eos_token_id=tokenizer.eos_token_id,
                       temperature=args.temperature)
        reply = tokenizer.decode(out[0].tolist())[len(prompt):].strip()
        history.append((user, reply))
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
