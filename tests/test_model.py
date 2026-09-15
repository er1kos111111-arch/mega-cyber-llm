"""Model tests: init, parameter count, forward, backward, loss, KV cache."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.architecture import MCLLM
from model.config import ModelConfig


def test_parameter_count_matches_formula(tiny_config, model):
    assert model.num_parameters() == tiny_config.total_parameters()


def test_all_weights_from_scratch(model):
    # every parameter must be a leaf nn.Parameter initialized from random/normal
    for name, p in model.named_parameters():
        assert p.requires_grad, f"{name} should require grad"
        assert torch.isfinite(p).all()


def test_forward_shape(model, tiny_config):
    b, s = 2, 16
    ids = torch.randint(0, tiny_config.vocab_size, (b, s))
    logits = model(ids)
    assert logits.shape == (b, s, tiny_config.vocab_size)


def test_forward_loss_finite(model, tiny_config):
    ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    loss = model.forward_loss(ids)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_backward_flows_gradients(model, tiny_config):
    ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    loss = model.forward_loss(ids)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_causal_attention_is_causal(model, tiny_config):
    # output for token t must not depend on tokens after t
    ids = torch.randint(0, tiny_config.vocab_size, (1, 16))
    base = model(ids)
    modified = ids.clone()
    modified[0, -1] = (modified[0, -1] + 1) % tiny_config.vocab_size
    out = model(modified)
    # all positions except the last are identical
    assert torch.allclose(base[:, :-1], out[:, :-1], atol=1e-6)


def test_kv_cache_matches_full_forward(model, tiny_config):
    from model.attention import KVCache
    b, s = 1, 8
    ids = torch.randint(0, tiny_config.vocab_size, (b, s))
    full = model(ids)

    cache = [
        KVCache(1, tiny_config.num_kv_heads, tiny_config.head_dim,
                tiny_config.max_position_embeddings, batch_size=b)
        for _ in range(tiny_config.num_layers)
    ]
    pos = torch.arange(s).unsqueeze(0)
    cached = model(ids, position_ids=pos, kv_cache=cache)
    assert torch.allclose(full, cached, atol=1e-5)


def test_gradient_checkpointing_friendly(model):
    # model should have named layers for potential gradient checkpointing
    assert hasattr(model, "layers")
    assert len(model.layers) == model.config.num_layers


def test_gradient_checkpointing_matches_full(model, tiny_config):
    torch.manual_seed(0)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    full = model.forward_loss(ids)

    model.enable_gradient_checkpointing()
    ckpt = model.forward_loss(ids)
    assert torch.allclose(full, ckpt, atol=1e-5)
    ckpt.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
