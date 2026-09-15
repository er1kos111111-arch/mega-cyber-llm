"""Checkpoint tests: save, load, resume, integrity."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.architecture import MCLLM
from model.config import RunConfig
from training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    validate_checkpoint,
)
from training.optimizer import AdamW


def _make_config(tokenizer):
    from model.config import ModelConfig, TrainingConfig
    return RunConfig(
        model=ModelConfig(vocab_size=len(tokenizer), hidden_size=32, num_layers=2,
                          num_attention_heads=4, num_kv_heads=2, intermediate_size=64,
                          max_position_embeddings=64),
        training=TrainingConfig(),
        out_dir="",  # overridden per-test
    )


def test_save_and_load_roundtrip(tiny_config, tokenizer, tmp_path):
    torch.manual_seed(0)
    model = MCLLM(tiny_config)
    out = str(tmp_path)

    # do a few optimizer steps to create a non-trivial state
    opt = AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
        loss = model.forward_loss(ids)
        opt.zero_grad()
        loss.backward()
        opt.step()

    weights_before = {k: v.clone() for k, v in model.state_dict().items()}

    cfg = _make_config(tokenizer)
    cfg.out_dir = out
    meta = save_checkpoint(model, opt, None, step=100, out_dir=out, config=cfg,
                           tokens_seen=1234, loss=3.14)

    # fresh model, load checkpoint
    model2 = MCLLM(tiny_config)
    step, meta2 = load_checkpoint(model2, out, step=100)

    assert step == 100
    assert meta2["tokens_seen"] == 1234
    for k, v in weights_before.items():
        assert torch.allclose(model2.state_dict()[k], v, atol=1e-6)

    assert validate_checkpoint(os.path.join(out, "mc_llm_step_000100_meta.json"))


def test_resume_from_latest(tiny_config, tokenizer, tmp_path):
    torch.manual_seed(1)
    model = MCLLM(tiny_config)
    out = str(tmp_path)
    cfg = _make_config(tokenizer)
    cfg.out_dir = out
    save_checkpoint(model, None, None, step=50, out_dir=out, config=cfg)
    save_checkpoint(model, None, None, step=100, out_dir=out, config=cfg)

    model2 = MCLLM(tiny_config)
    step, _ = load_checkpoint(model2, out)  # no step -> latest
    assert step == 100


def test_sharded_checkpoint(tiny_config, tokenizer, tmp_path):
    torch.manual_seed(2)
    model = MCLLM(tiny_config)
    out = str(tmp_path)
    cfg = _make_config(tokenizer)
    cfg.out_dir = out
    # tiny shard size to force splitting
    meta = save_checkpoint(model, None, None, step=10, out_dir=out, config=cfg,
                           shard_size=1024)

    model2 = MCLLM(tiny_config)
    step, _ = load_checkpoint(model2, out, step=10)
    assert step == 10
    for k, v in model.state_dict().items():
        assert torch.allclose(model2.state_dict()[k], v, atol=1e-6)
