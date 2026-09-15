"""Evaluation entry point for MC-LLM.

Computes:

* language-modeling loss + perplexity on a held-out token stream;
* generation quality on built-in benchmarks;
* repetition / distinctness diagnostics.

Writes a JSON report of the form::

    {
      "checkpoint": "...",
      "parameters": 0,
      "validation_loss": 0.0,
      "perplexity": 0.0,
      "tokens_seen": 0
    }
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch

from inference.generate import generate
from inference.loader import load_model
from evaluation import benchmarks, metrics


def compute_validation_loss(model, data_dir, tokenizer, seq_len=None, max_batches=50,
                            batch_size=8, device="cpu") -> float:
    from data.dataset import ShardedTokenDataset
    import torch.nn.functional as F
    if seq_len is None:
        seq_len = model.config.max_position_embeddings
    dataset = ShardedTokenDataset(data_dir, seq_len, shuffle_shards=False, seed=0)
    total = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for seq in dataset:
            if n >= max_batches:
                break
            # build input/target
            x = seq[:-1].unsqueeze(0).to(device)
            y = seq[1:].unsqueeze(0).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total += loss.item()
            n += 1
    return total / max(1, n)


def run_benchmarks(model, tokenizer, suite: str, device: str,
                   max_new_tokens: int = 32) -> Dict:
    entries = benchmarks.get_benchmark(suite)
    results = []
    score = 0.0
    scored = 0
    for entry in entries:
        prompt = entry["prompt"]
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = generate(model, ids, max_new_tokens=max_new_tokens,
                       eos_token_id=tokenizer.eos_token_id, temperature=0.0)
        text = tokenizer.decode(out[0].tolist())
        answer = text[len(prompt):].strip()
        ok = 0.0
        if entry.get("answers"):
            ok = metrics.contains_any(answer, entry["answers"])
            score += ok
            scored += 1
        results.append({"prompt": prompt, "completion": answer, "match": bool(ok),
                        "type": entry["type"]})
        rep = metrics.repetition_ratio(out[0].tolist())
        results[-1]["repetition"] = round(rep, 4)
    return {"results": results, "accuracy": round(score / max(1, scored), 4),
            "scored": scored, "total": len(entries)}


def evaluate(
    checkpoint_dir: str,
    tokenizer_path: str = "tokenizer/tokenizer_config.json",
    data_dir: str = "data/shards",
    suite: str = "all",
    out_json: str = "evaluation_report.json",
    device: str = "cpu",
):
    model, tokenizer, meta = load_model(checkpoint_dir, tokenizer_path, device=device)
    n_params = sum(p.numel() for p in model.parameters())

    report = {
        "checkpoint": checkpoint_dir,
        "parameters": n_params,
        "tokens_seen": meta.get("tokens_seen", 0),
        "validation_loss": None,
        "perplexity": None,
    }

    if os.path.exists(os.path.join(data_dir, "shards_manifest.json")):
        val_loss = compute_validation_loss(model, data_dir, tokenizer, device=device)
        report["validation_loss"] = round(val_loss, 4)
        report["perplexity"] = round(metrics.perplexity(val_loss), 2)

    bench = run_benchmarks(model, tokenizer, suite, device)
    report["benchmarks"] = bench

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "benchmarks"},
                     ensure_ascii=False, indent=2))
    print(f"benchmark accuracy ({suite}): {bench['accuracy']}")
    print(f"report written to {out_json}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate MC-LLM")
    parser.add_argument("--checkpoint", default="checkpoints")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer_config.json")
    parser.add_argument("--data", default="data/shards")
    parser.add_argument("--suite", default="all", choices=benchmarks.list_benchmarks())
    parser.add_argument("--out", default="evaluation_report.json")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.tokenizer, args.data, args.suite, args.out)


if __name__ == "__main__":
    main()
