"""Generation benchmark + quality report for MC-LLM.

Runs a fixed set of prompts through a checkpoint, measures quality metrics
(repetition, invalid chars, coherence), and writes a JSON report plus the
sample generations.  Use it after each important checkpoint to see whether
the model is actually improving.

    python scripts/quality_report.py --checkpoint checkpoints_sft --out eval_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from inference.generate import generate, CHAT_PRESET
from evaluation.quality_metrics import assess_generation, is_broken

PROMPTS = [
    "Привет",
    "Как дела?",
    "Мне сегодня скучно.",
    "Что можно делать вечером?",
    "Я сегодня устал.",
    "Кстати, что нового?",
    "А ты любишь игры?",
    "Мне нужно с кем-нибудь поговорить.",
]


def run_benchmark(checkpoint: str, tokenizer_path: str, device: str,
                  prompts=None, max_new_tokens: int = 60) -> dict:
    from inference.loader import load_model
    model, tokenizer, meta = load_model(checkpoint, tokenizer_path, device=device)
    prompts = prompts or PROMPTS

    results = []
    broken = 0
    total_rep = 0.0
    for p in prompts:
        ids = tokenizer.tokenize_chat([{"role": "user", "content": p}],
                                      add_generation_prompt=True)
        prompt_len = len(ids)
        in_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate(model, in_ids, max_new_tokens=max_new_tokens,
                       eos_token_id=tokenizer.eos_token_id,
                       temperature=CHAT_PRESET["temperature"],
                       top_p=CHAT_PRESET["top_p"], top_k=CHAT_PRESET["top_k"],
                       repetition_penalty=CHAT_PRESET["repetition_penalty"],
                       frequency_penalty=CHAT_PRESET["frequency_penalty"],
                       presence_penalty=CHAT_PRESET["presence_penalty"], seed=1)
        new_ids = out[0].tolist()[prompt_len:]
        reply = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        m = assess_generation(reply, new_ids)
        m["prompt"] = p
        m["reply"] = reply
        m["broken"] = is_broken(m)
        broken += int(m["broken"])
        total_rep += m["repetition_ratio"]
        results.append(m)

    return {
        "checkpoint": checkpoint,
        "parameters": sum(x.numel() for x in model.parameters()),
        "samples": results,
        "broken_rate": round(broken / max(1, len(results)), 3),
        "avg_repetition": round(total_rep / max(1, len(results)), 3),
        "avg_length": round(sum(r["length"] for r in results) / max(1, len(results)), 1),
    }


def print_report(report: dict):
    print("=" * 60)
    print("MEGA-CYBER LLM — GENERATION QUALITY")
    print("=" * 60)
    print(f"checkpoint:  {report['checkpoint']}")
    print(f"parameters:  {report['parameters']:,}")
    print(f"broken rate: {report['broken_rate']}")
    print(f"avg repeat:  {report['avg_repetition']}")
    print(f"avg length:  {report['avg_length']}")
    print("-" * 60)
    for s in report["samples"]:
        flag = " [BROKEN]" if s["broken"] else ""
        print(f"Q: {s['prompt']}")
        print(f"A: {s['reply']}{flag}")
        print()
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description="MC-LLM generation quality report")
    p.add_argument("--checkpoint", default="checkpoints_sft")
    p.add_argument("--tokenizer", default="tokenizer/tokenizer_config.json")
    p.add_argument("--out", default="evaluation/quality_report.json")
    p.add_argument("--max-new-tokens", type=int, default=60)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = run_benchmark(args.checkpoint, args.tokenizer, device,
                           max_new_tokens=args.max_new_tokens)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print_report(report)
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
