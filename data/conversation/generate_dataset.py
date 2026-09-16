"""Streaming conversational dataset generator for MC-LLM.

The main driver: generate → validate → deduplicate → buffer → flush shard →
clear memory → repeat, while monitoring resources and supporting resume.

It runs until a safe resource limit is reached (RAM / disk / time / token
budget), producing many shard files:

    data/conversation/shards/shard_000000.jsonl
    data/conversation/shards/shard_000000.meta.json
    ...

Each JSONL line is a full multi-turn conversation plus metadata.  Token
counts are computed with the real CyberTokenizer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.resource_monitor import ResourceMonitor, ResourceLimits, format_snapshot
from data.conversation.generator import DialogueGenerator, GenConfig
from data.conversation import quality
from data.conversation.dedup import ConversationDedup

SHARDS_DIR = "data/conversation/shards"
PROGRESS_FILE = "progress.json"


def ensure_tokenizer(tokenizer_dir: str, generator: DialogueGenerator, vocab_size: int = 8192,
                     sample_dialogues: int = 8000):
    """Return a real CyberTokenizer, training one on a large sample of
    *generated conversations* (not just the raw phrase bank) if needed."""
    from tokenizer.tokenizer import CyberTokenizer
    cfg_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    if os.path.exists(cfg_path):
        return CyberTokenizer(cfg_path)

    from tokenizer.trainer import CyberTokenizerTrainer
    from tokenizer.vocab import (
        SPECIAL_TOKENS, NUM_BYTE_TOKENS, build_special_token_map,
        save_vocab, save_merges,
    )
    os.makedirs(tokenizer_dir, exist_ok=True)

    # generate a diverse sample of conversations for tokenizer training
    print(f"[gen] generating {sample_dialogues:,} dialogues to train the tokenizer...")
    texts = [generator.generate()["text"] for _ in range(sample_dialogues)]
    texts.append(generator.dump_phrase_text())  # ensure base phrases are covered

    trainer = CyberTokenizerTrainer(vocab_size=vocab_size, min_pair_frequency=1)
    merges = trainer.train(texts)
    total = NUM_BYTE_TOKENS + len(SPECIAL_TOKENS) + len(merges)
    save_vocab(os.path.join(tokenizer_dir, "vocab.json"),
               build_special_token_map(), merges, total)
    save_merges(os.path.join(tokenizer_dir, "merges.json"), merges)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"name": "CyberTokenizer", "version": "1.0", "vocab_size": total,
                   "special_tokens": SPECIAL_TOKENS, "vocab_file": "vocab.json",
                   "merges_file": "merges.json", "model_max_length": 8192}, f,
                  ensure_ascii=False, indent=2)
    print(f"[gen] trained tokenizer on phrase bank: vocab={total}")
    return CyberTokenizer(cfg_path)


def _scan_shards(out_dir: str):
    """Return (last_index, total_dialogues, total_turns, total_tokens, size_mb)."""
    if not os.path.isdir(out_dir):
        return -1, 0, 0, 0, 0.0
    last = -1
    td = tt = tk = 0
    size = 0.0
    for name in os.listdir(out_dir):
        if name.endswith(".meta.json"):
            try:
                idx = int(name.split("_")[1].split(".")[0])
                meta = json.load(open(os.path.join(out_dir, name), encoding="utf-8"))
                last = max(last, idx)
                td += meta.get("dialogues", 0)
                tt += meta.get("turns", 0)
                tk += meta.get("tokens", 0)
                size += meta.get("size_mb", 0.0)
            except Exception:
                continue
    return last, td, tt, tk, size


def run(
    out_dir: str = SHARDS_DIR,
    tokenizer_dir: str = "tokenizer",
    seed: int = 0,
    vocab_size: int = 8192,
    dialogues_per_shard: int = 2000,
    tokens_per_shard: int = 2_000_000,
    log_every: int = 200,
    limits: ResourceLimits = None,
    length_dist=None,
    template_max_repeat: int = 3,
) -> dict:
    limits = limits or ResourceLimits()
    monitor = ResourceMonitor(limits)

    gen_cfg = GenConfig()
    if length_dist is not None:
        gen_cfg.length_dist = length_dist
    generator = DialogueGenerator(seed=seed, config=gen_cfg)
    dedup = ConversationDedup(template_max_repeat=template_max_repeat)
    tokenizer = ensure_tokenizer(tokenizer_dir, generator, vocab_size)

    os.makedirs(out_dir, exist_ok=True)

    # resume
    last_idx, total_d, total_turns, total_tokens, total_mb = _scan_shards(out_dir)
    start_idx = last_idx + 1
    print(f"[gen] resume from shard {start_idx} "
          f"(already have {total_d:,} dialogues, {total_tokens:,} tokens)")

    # per-shard buffers
    buffer_lines = []
    buffer_dialogues = 0
    buffer_tokens = 0

    lang_counts = {"ru": 0, "en": 0, "mixed": 0}
    bucket_counts = {}
    total_generated = 0
    total_rejected = 0

    def flush_shard(idx, lines, dialogues, tokens):
        if not lines:
            return
        shard_path = os.path.join(out_dir, f"shard_{idx:06d}.jsonl")
        with open(shard_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        size_mb = os.path.getsize(shard_path) / 1e6
        meta = {"shard": idx, "dialogues": dialogues, "turns": int(sum(
            json.loads(l)["turns"] for l in lines)), "tokens": tokens,
            "size_mb": round(size_mb, 3)}
        with open(os.path.join(out_dir, f"shard_{idx:06d}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta

    start_time = time.time()
    snap = monitor.snapshot()
    idx = start_idx

    print("=" * 60)
    print("Generating conversational dataset (streaming)...")
    print("=" * 60)

    while True:
        d = generator.generate()
        total_generated += 1

        if not quality.check_dialogue(d["text"]):
            total_rejected += 1
            continue

        keep, _reason = dedup.should_keep(d)
        if not keep:
            continue

        dedup.record(d)
        n_tokens = len(tokenizer.encode(d["text"]))

        line = json.dumps({"text": d["text"], "turns": d["turns"],
                           "lang": d["lang"], "signature": list(d["signature"])},
                          ensure_ascii=False)
        buffer_lines.append(line)
        buffer_dialogues += 1
        buffer_tokens += n_tokens

        total_d += 1
        total_turns += d["turns"]
        total_tokens += n_tokens
        lang_counts[d["lang"]] = lang_counts.get(d["lang"], 0) + 1
        bucket_counts[d["bucket"]] = bucket_counts.get(d["bucket"], 0) + 1

        # periodic log + resource check
        if total_generated % log_every == 0:
            snap = monitor.snapshot()
            print(f"  gen {total_generated:,} | kept {total_d:,} | "
                  f"tokens {total_tokens:,} | {format_snapshot(snap)}")

        # flush conditions
        do_flush = (buffer_dialogues >= dialogues_per_shard
                    or buffer_tokens >= tokens_per_shard
                    or monitor.should_flush(snap))
        if do_flush and buffer_lines:
            meta = flush_shard(idx, buffer_lines, buffer_dialogues, buffer_tokens)
            total_mb += meta["size_mb"]
            buffer_lines, buffer_dialogues, buffer_tokens = [], 0, 0
            idx += 1
            dedup.release_session_memory()
            if meta:
                print(f"[gen] flushed shard {meta['shard']} "
                      f"({meta['dialogues']:,} dial, {meta['tokens']:,} tok, {meta['size_mb']} MB)")

        # stop check
        snap = monitor.snapshot()
        stop, reason = monitor.should_stop(snap, dialogues=total_d,
                                           tokens=total_tokens, shards=idx - start_idx)
        if stop:
            print(f"[gen] stopping: {reason}")
            break

    # final flush
    if buffer_lines:
        meta = flush_shard(idx, buffer_lines, buffer_dialogues, buffer_tokens)
        if meta:
            total_mb += meta["size_mb"]
            idx += 1

    elapsed = time.time() - start_time
    report = {
        "dialogues": total_d,
        "turns": total_turns,
        "tokens": total_tokens,
        "shards": idx - start_idx,
        "size_mb": round(total_mb, 2),
        "avg_turns": round(total_turns / max(1, total_d), 2),
        "avg_dialogue_tokens": round(total_tokens / max(1, total_d), 1),
        "lang": lang_counts,
        "buckets": bucket_counts,
        "dedup": dict(dedup.stats),
        "rejected": total_rejected,
        "generated": total_generated,
        "ram_peak_gb": round(monitor.peak_ram_gb, 2),
        "runtime_seconds": round(elapsed, 1),
        "tokenizer_vocab": len(tokenizer),
    }
    # persist progress for resume
    with open(os.path.join(out_dir, PROGRESS_FILE), "w", encoding="utf-8") as f:
        json.dump({"report": report, "last_shard": idx - 1}, f, ensure_ascii=False, indent=2)
    return report


def print_report(report: dict):
    print()
    print("=" * 60)
    print("CONVERSATIONAL DATASET REPORT")
    print("=" * 60)
    print(f"Total dialogues:        {report['dialogues']:,}")
    print(f"Total turns:            {report['turns']:,}")
    print(f"Total tokens:           {report['tokens']:,}")
    print(f"Total shards:           {report['shards']}")
    print(f"Dataset size:           {report['size_mb']} MB")
    print()
    print(f"Average dialogue length:{report['avg_turns']} turns "
          f"({report['avg_dialogue_tokens']} tokens)")
    print()
    print(f"Russian:                {report['lang'].get('ru', 0):,}")
    print(f"English:                {report['lang'].get('en', 0):,}")
    print(f"Mixed:                  {report['lang'].get('mixed', 0):,}")
    print()
    for b, c in sorted(report["buckets"].items()):
        print(f"  {b:<12}: {c:,}")
    print()
    print(f"Exact duplicates removed:   {report['dedup'].get('exact', 0):,}")
    print(f"Near duplicates removed:    {report['dedup'].get('near', 0):,}")
    print(f"Template duplicates removed:{report['dedup'].get('template', 0):,}")
    print(f"Low-quality removed:        {report['rejected']:,}")
    print()
    print(f"RAM peak:               {report['ram_peak_gb']} GB")
    print(f"Runtime:                {report['runtime_seconds']/60:.1f} min")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description="Generate conversational dataset")
    p.add_argument("--out", default=SHARDS_DIR)
    p.add_argument("--tokenizer", default="tokenizer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--dialogues-per-shard", type=int, default=2000)
    p.add_argument("--tokens-per-shard", type=int, default=2_000_000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--ram-fraction", type=float, default=0.85)
    p.add_argument("--disk-limit-gb", type=float, default=8.0)
    p.add_argument("--max-runtime-seconds", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--max-dialogues", type=int, default=0)
    p.add_argument("--template-max-repeat", type=int, default=3)
    args = p.parse_args()

    limits = ResourceLimits(
        ram_fraction=args.ram_fraction,
        disk_limit_gb=args.disk_limit_gb,
        max_runtime_seconds=args.max_runtime_seconds,
        max_tokens=args.max_tokens,
        max_dialogues=args.max_dialogues,
    )
    report = run(
        out_dir=args.out, tokenizer_dir=args.tokenizer, seed=args.seed,
        vocab_size=args.vocab_size, dialogues_per_shard=args.dialogues_per_shard,
        tokens_per_shard=args.tokens_per_shard, log_every=args.log_every,
        limits=limits, template_max_repeat=args.template_max_repeat,
    )
    print_report(report)


if __name__ == "__main__":
    main()
