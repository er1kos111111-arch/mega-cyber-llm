"""Generation CLI for MC-LLM.

    python scripts/generate.py --checkpoint checkpoints --prompt "Привет!"

Supports greedy / temperature / top-k / top-p / repetition-penalty / min-p,
stop tokens, max tokens, and (optional) streaming output.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch

from inference.generate import generate, generate_stream
from inference.loader import load_model


def main():
    parser = argparse.ArgumentParser(description="Generate with MC-LLM")
    parser.add_argument("--checkpoint", default="checkpoints")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer_config.json")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, meta = load_model(args.checkpoint, args.tokenizer, device=device)

    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)

    print(args.prompt, end="", flush=True)

    if args.stream:
        prev = len(ids[0])
        for seq in generate_stream(
            model, ids, max_new_tokens=args.max_tokens,
            eos_token_id=tokenizer.eos_token_id, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty, min_p=args.min_p,
            seed=args.seed, use_cache=not args.no_cache):
            new = seq[0][prev:]
            print(tokenizer.decode(new.tolist()), end="", flush=True)
            prev = len(seq[0])
        print()
    else:
        out = generate(
            model, ids, max_new_tokens=args.max_tokens,
            eos_token_id=tokenizer.eos_token_id, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty, min_p=args.min_p,
            seed=args.seed, use_cache=not args.no_cache)
        print(tokenizer.decode(out[0].tolist())[len(args.prompt):])


if __name__ == "__main__":
    main()
