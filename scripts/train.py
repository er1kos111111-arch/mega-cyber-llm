"""Training entry point for MC-LLM.

    python scripts/train.py --config configs/100m.yaml

Supports distributed training via torchrun (DDP/FSDP); for a single GPU/CPU
just run directly.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os

from model.config import RunConfig
from training import train as train_module
from training import distributed as dist


def main():
    parser = argparse.ArgumentParser(description="Train MC-LLM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    config = RunConfig.from_yaml(args.config)
    os.makedirs(config.out_dir, exist_ok=True)

    # optional distributed init (no-op when running single-process)
    world_size = config.distributed.world_size
    if world_size > 1:
        dist.init_process_group(config.distributed.backend)

    from data.dataset import build_dataloader

    seq_len = config.training.sequence_length
    micro = config.training.micro_batch_size

    # train/val split across shards: reuse the same dir but shuffle seed
    train_loader = build_dataloader(
        config.data_dir, seq_len, micro, num_workers=0,
        seed=config.training.seed)
    eval_loader = build_dataloader(
        config.data_dir, seq_len, micro, num_workers=0,
        seed=config.training.seed + 1)

    summary = train_module.train(
        config, train_loader=train_loader, eval_loader=eval_loader,
        resume=not args.no_resume)

    print("=" * 52)
    print("Training complete")
    print(f"  parameters: {summary['parameters']:,}")
    print(f"  final step: {summary['final_step']}")
    print(f"  tokens:     {summary['tokens_seen']:,}")
    print(f"  loss:       {summary['loss']:.4f}")
    print(f"  checkpoint: {summary['checkpoint']}")
    print("=" * 52)


if __name__ == "__main__":
    main()
