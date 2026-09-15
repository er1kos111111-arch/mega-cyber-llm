"""Shared model/tokenizer loading for inference, chat and evaluation."""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import torch

from model.architecture import MCLLM
from model.config import ModelConfig
from tokenizer.tokenizer import CyberTokenizer
from training.checkpoint import load_checkpoint


def load_tokenizer(tokenizer_path: str) -> CyberTokenizer:
    if os.path.isdir(tokenizer_path):
        tokenizer_path = os.path.join(tokenizer_path, "tokenizer_config.json")
    return CyberTokenizer(tokenizer_path)


def load_model(
    checkpoint_dir: str,
    tokenizer_path: str = "tokenizer/tokenizer_config.json",
    step: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[MCLLM, CyberTokenizer, dict]:
    """Load a trained MC-LLM from a checkpoint directory.

    Returns (model, tokenizer, meta).  The checkpoint metadata contains the
    full model config, so no separate config file is required.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    meta_path = os.path.join(checkpoint_dir, "latest.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No 'latest.json' in {checkpoint_dir}; train first.")

    with open(meta_path, "r", encoding="utf-8") as f:
        latest = json.load(f)

    cfg = latest.get("config", {})
    model_cfg = ModelConfig.from_dict(cfg.get("model", cfg))
    model = MCLLM(model_cfg)

    load_step = step if step is not None else latest["step"]
    _, meta = load_checkpoint(model, checkpoint_dir, step=load_step, map_location="cpu")
    model = model.to(device)
    model.eval()

    tokenizer = load_tokenizer(tokenizer_path)
    return model, tokenizer, meta
