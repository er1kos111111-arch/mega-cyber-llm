"""End-to-end data preparation for MC-LLM.

    raw → clean → filter/dedup → tokenize → shards

Usage:
    python scripts/prepare_data.py --input data/raw --tokenizer tokenizer \
        --out data/shards --lang-ratio ru:0.5,en:0.5
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

from data.cleaner import clean_documents
from data.dedup import exact_dedup
from data.filter import detect_language, filter_document
from data.shard import read_texts, shard_texts
from tokenizer.tokenizer import CyberTokenizer


def iter_raw(paths) -> Iterator[str]:
    for path in paths:
        if os.path.isdir(path):
            for ext in ("*.txt", "*.jsonl", "*.md", "*.html"):
                for f in glob.glob(os.path.join(path, "**", ext), recursive=True):
                    yield from read_texts(f)
        else:
            yield from read_texts(path)


def main():
    parser = argparse.ArgumentParser(description="Prepare MC-LLM data")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer_config.json")
    parser.add_argument("--out", default="data/shards")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument("--max-chars", type=int, default=100000)
    parser.add_argument("--lang-ratio", default=None,
                        help="comma-separated lang:ratio, e.g. ru:0.5,en:0.5")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--tokens-per-shard", type=int, default=1_000_000)
    parser.add_argument("--max-docs", type=int, default=0)
    args = parser.parse_args()

    tok = CyberTokenizer(args.tokenizer)

    lang_ratio = None
    if args.lang_ratio:
        lang_ratio = {}
        for pair in args.lang_ratio.split(","):
            k, v = pair.split(":")
            lang_ratio[k.strip()] = float(v.strip())

    os.makedirs(args.out, exist_ok=True)

    print("[prepare_data] cleaning + filtering...")
    kept = 0
    total = 0
    lang_counts = {"ru": 0, "en": 0, "other": 0}

    def cleaned_stream():
        nonlocal kept, total
        for doc in iter_raw(args.input):
            if args.max_docs and total >= args.max_docs:
                break
            total += 1
            cleaned = next(clean_documents([doc]), "")
            if not cleaned:
                continue
            if not filter_document(cleaned, min_chars=args.min_chars,
                                   max_chars=args.max_chars):
                continue
            lang = detect_language(cleaned)
            if lang_ratio is not None:
                # enforce language ratio by down-sampling
                import random
                ratio = lang_ratio.get(lang, 0.0)
                if random.random() > ratio:
                    continue
            lang_counts[lang] += 1
            kept += 1
            yield cleaned

    print("[prepare_data] deduplicating...")
    deduped = exact_dedup(cleaned_stream())

    print("[prepare_data] tokenizing + sharding...")
    manifest = shard_texts(deduped, tok, args.out, args.vocab_size,
                           tokens_per_shard=args.tokens_per_shard,
                           max_seq_len=args.max_seq_len)

    print(f"[prepare_data] done. docs_total={total}, kept={kept}, "
          f"langs={lang_counts}")
    print(f"[prepare_data] manifest: {manifest}")


if __name__ == "__main__":
    main()
