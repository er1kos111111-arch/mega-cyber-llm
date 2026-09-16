"""Build a REAL training dataset for MC-LLM from open Hugging Face data.

Pipeline:
    download (Wikipedia ru/en + PersonaChat dialogues)
    -> train CyberTokenizer on the real data (32k vocab)
    -> clean + filter + dedup + shard
    -> data/shards/ (ready for training)

Usage:
    python scripts/build_real_dataset.py --max-rows 200000
    python scripts/train.py --config configs/100m.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_real_data(raw_dir: str, max_rows: int, max_files: int,
                       include_english: bool = False) -> dict:
    from data.download import download_preset
    stats = {}
    # Russian primary; English is opt-in (small fraction of the Russian amount)
    presets = [("wikipedia_ru", 1.0), ("persona_chat_ru", 1.0)]
    if include_english:
        presets.append(("wikipedia_en", 0.15))
    for name, frac in presets:
        try:
            rows = int(max_rows * frac) if max_rows else 0
            n = download_preset(name, out_dir=raw_dir, max_rows=rows,
                                max_files=max_files)
            stats[name] = n
            print(f"[real-data] {name}: {n:,} docs")
        except Exception as e:
            print(f"[real-data] {name}: SKIPPED ({e})")
            stats[name] = 0
    return stats


def train_tokenizer(raw_dir: str, out_dir: str, vocab_size: int,
                    max_chars: int = 2_000_000) -> int:
    from scripts.train_tokenizer import iter_corpus
    from tokenizer.trainer import CyberTokenizerTrainer
    from tokenizer.vocab import (
        SPECIAL_TOKENS, NUM_BYTE_TOKENS, build_special_token_map,
        save_vocab, save_merges,
    )
    os.makedirs(out_dir, exist_ok=True)
    trainer = CyberTokenizerTrainer(vocab_size=vocab_size, min_pair_frequency=2)
    print(f"[real-data] training CyberTokenizer on a {max_chars/1e6:.0f}M-char "
          f"sample of real data...")

    def limited():
        total = 0
        for text in iter_corpus([raw_dir]):
            if total >= max_chars:
                break
            total += len(text)
            yield text

    merges = trainer.train(limited())
    total = NUM_BYTE_TOKENS + len(SPECIAL_TOKENS) + len(merges)
    save_vocab(os.path.join(out_dir, "vocab.json"), build_special_token_map(),
               merges, total)
    save_merges(os.path.join(out_dir, "merges.json"), merges)
    cfg = {"name": "CyberTokenizer", "version": "1.0", "vocab_size": total,
           "special_tokens": SPECIAL_TOKENS, "vocab_file": "vocab.json",
           "merges_file": "merges.json", "model_max_length": 8192}
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[real-data] tokenizer vocab: {total}")
    return total


def shard_real_data(raw_dir: str, tokenizer_dir: str, out_dir: str,
                    vocab_size: int, max_seq_len: int) -> str:
    from data.cleaner import clean_documents
    from data.dedup import exact_dedup
    from data.filter import filter_document
    from data.shard import shard_texts
    from tokenizer.tokenizer import CyberTokenizer
    from scripts.train_tokenizer import iter_corpus

    tok = CyberTokenizer(tokenizer_dir)
    os.makedirs(out_dir, exist_ok=True)
    print("[real-data] cleaning + sharding real data...")

    def stream():
        for doc in iter_corpus([raw_dir]):
            cleaned = next(clean_documents([doc]), "")
            if cleaned and filter_document(cleaned, min_chars=30):
                yield cleaned

    manifest = shard_texts(exact_dedup(stream()), tok, out_dir, vocab_size,
                           max_seq_len=max_seq_len)
    print(f"[real-data] shards ready: {manifest}")
    return manifest


def main():
    p = argparse.ArgumentParser(description="Build a real MC-LLM dataset")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--tokenizer-dir", default="tokenizer")
    p.add_argument("--out", default="data/shards")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--max-rows", type=int, default=0, help="max docs per dataset (0 = all)")
    p.add_argument("--max-files", type=int, default=0, help="max parquet files per dataset (0 = all)")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    if not args.skip_download:
        download_real_data(args.raw_dir, args.max_rows, args.max_files)

    vocab = train_tokenizer(args.raw_dir, args.tokenizer_dir, args.vocab_size)
    shard_real_data(args.raw_dir, args.tokenizer_dir, args.out, vocab,
                    args.max_seq_len)
    print("\n[real-data] done. Now train:")
    print(f"  python scripts/train.py --config configs/100m.yaml")


if __name__ == "__main__":
    main()
