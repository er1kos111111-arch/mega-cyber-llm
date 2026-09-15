"""Train CyberTokenizer from scratch on the project corpus.

Usage:
    python scripts/train_tokenizer.py --input data/clean --vocab-size 32768 --out tokenizer
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import json
import os
from typing import Iterator

from tokenizer.trainer import CyberTokenizerTrainer
from tokenizer.vocab import (
    SPECIAL_TOKENS,
    NUM_BYTE_TOKENS,
    build_special_token_map,
    save_vocab,
    save_merges,
)


def iter_corpus(input_paths) -> Iterator[str]:
    for path in input_paths:
        if os.path.isdir(path):
            for ext in ("*.txt", "*.jsonl", "*.md"):
                for f in glob.glob(os.path.join(path, "**", ext), recursive=True):
                    yield from _read_file(f)
        else:
            yield from _read_file(path)


def _read_file(path: str) -> Iterator[str]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if ext == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield obj["text"] if isinstance(obj, dict) and "text" in obj else str(obj)
        else:
            yield f.read()


def main():
    parser = argparse.ArgumentParser(description="Train CyberTokenizer")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-pair-frequency", type=int, default=2)
    parser.add_argument("--out", default="tokenizer")
    parser.add_argument("--max-chars", type=int, default=0,
                        help="limit training corpus chars (0 = unlimited)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    trainer = CyberTokenizerTrainer(
        vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
    )

    print("[CyberTokenizer] reading corpus...")
    def limited():
        total = 0
        for text in iter_corpus(args.input):
            if not text:
                continue
            if args.max_chars and total >= args.max_chars:
                break
            total += len(text)
            yield text
        print(f"[CyberTokenizer] corpus chars: {total:,}")

    merges = trainer.train(limited(), min_frequency=args.min_pair_frequency)

    total_vocab = NUM_BYTE_TOKENS + len(SPECIAL_TOKENS) + len(merges)
    special_map = build_special_token_map()

    save_vocab(os.path.join(args.out, "vocab.json"), special_map, merges, total_vocab)
    save_merges(os.path.join(args.out, "merges.json"), merges)

    cfg = {
        "name": "CyberTokenizer",
        "version": "1.0",
        "model_max_length": 8192,
        "vocab_size": total_vocab,
        "special_tokens": SPECIAL_TOKENS,
        "vocab_file": "vocab.json",
        "merges_file": "merges.json",
        "bos_token": "<BOS>", "eos_token": "<EOS>",
        "pad_token": "<PAD>", "unk_token": "<UNK>",
    }
    with open(os.path.join(args.out, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[CyberTokenizer] done: vocab_size={total_vocab}, "
          f"merges={len(merges)}")


if __name__ == "__main__":
    main()
