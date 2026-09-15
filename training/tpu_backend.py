"""JAX / TPU backend for MC-LLM (Google TPU Research Cloud).

A from-scratch JAX re-implementation of the MC-LLM architecture for Google
TPU.  It mirrors the PyTorch implementation in ``model/`` and ``training/``
so a config trained on either backend is semantically identical.  Everything
is initialized from scratch.

It supports:

* data-parallel training across all TPU cores (``jax.pmap``);
* loading the same binary shards produced by ``scripts/prepare_data.py``;
* from-scratch AdamW, gradient clipping, warmup+cosine LR;
* checkpoint save/load (with resume);
* greedy generation for a quick smoke test.

Install on a TPU VM (not needed for the local PyTorch path):

    pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

Run:

    python -m training.tpu_backend --config configs/100m.yaml --data-dir data/shards
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    _HAS_JAX = True
except ImportError:  # pragma: no cover - JAX not installed on this host
    _HAS_JAX = False


# ---------------------------------------------------------------------------
# config (mirrors model/config.py)
# ---------------------------------------------------------------------------
@dataclass
class JaxModelConfig:
    vocab_size: int = 32768
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    intermediate_size: int = 2048
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def total_parameters(self) -> int:
        d, hd = self.hidden_size, self.head_dim
        emb = self.vocab_size * d
        attn = d * d + 2 * self.num_kv_heads * hd * d + d * d
        ffn = 3 * d * self.intermediate_size
        return emb + self.num_layers * (attn + ffn) + self.vocab_size * d

    @classmethod
    def from_yaml(cls, path: str) -> "JaxModelConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        m = data.get("model", {})
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in m.items() if k in known})


# ---------------------------------------------------------------------------
# model (pure functional, parameters are a pytree)
# ---------------------------------------------------------------------------
def init_params(cfg: JaxModelConfig, key) -> Dict:
    d, hd = cfg.hidden_size, cfg.head_dim
    nk = cfg.num_kv_heads
    embeddings = random.normal(key, (cfg.vocab_size, d), dtype=jnp.float32) * (1.0 / math.sqrt(d))
    blocks = {}
    for i in range(cfg.num_layers):
        keys = random.split(random.fold_in(key, i), 9)
        blocks[f"block_{i}"] = {
            "attn_norm": jnp.ones(d),
            "ffn_norm": jnp.ones(d),
            "q": random.normal(keys[0], (cfg.num_heads * hd, d)) * 0.02,
            "k": random.normal(keys[1], (nk * hd, d)) * 0.02,
            "v": random.normal(keys[2], (nk * hd, d)) * 0.02,
            "o": random.normal(keys[3], (d, cfg.num_heads * hd)) * 0.02,
            "gate": random.normal(keys[4], (cfg.intermediate_size, d)) * 0.02,
            "up": random.normal(keys[5], (cfg.intermediate_size, d)) * 0.02,
            "down": random.normal(keys[6], (d, cfg.intermediate_size)) * 0.02,
        }
    return {
        "embeddings": embeddings,
        "blocks": blocks,
        "final_norm": jnp.ones(d),
        "lm_head": random.normal(key, (d, cfg.vocab_size)) * 0.02,
    }


def _rms_norm(x, weight, eps=1e-5):
    x = x.astype(jnp.float32)
    variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + eps) * weight).astype(x.dtype)


def _precompute_rope(cfg: JaxModelConfig):
    inv_freq = 1.0 / (cfg.rope_theta ** (jnp.arange(0, cfg.head_dim, 2) / cfg.head_dim))
    pos = jnp.arange(cfg.max_seq_len)
    freqs = jnp.outer(pos, inv_freq)
    return jnp.cos(freqs), jnp.sin(freqs)


def _apply_rope(x, cos, sin):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def forward(params: Dict, tokens: jnp.ndarray, cfg: JaxModelConfig,
            cos, sin) -> jnp.ndarray:
    b, s = tokens.shape
    x = params["embeddings"][tokens]
    pos = jnp.arange(s)
    c = cos[pos][None, :, None, :]
    sn = sin[pos][None, :, None, :]
    d, hd = cfg.hidden_size, cfg.head_dim
    nk = cfg.num_kv_heads
    qg = cfg.num_heads // nk

    for i in range(cfg.num_layers):
        blk = params["blocks"][f"block_{i}"]
        residual = x
        h = _rms_norm(x, blk["attn_norm"], cfg.rms_norm_eps)

        q = (h @ blk["q"].T).reshape(b, s, cfg.num_heads, hd)
        k = (h @ blk["k"].T).reshape(b, s, nk, hd)
        v = (h @ blk["v"].T).reshape(b, s, nk, hd)
        q = _apply_rope(q, c, sn)
        k = _apply_rope(k, c, sn)
        k = jnp.repeat(k, qg, axis=2)
        v = jnp.repeat(v, qg, axis=2)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(hd)
        mask = jnp.triu(jnp.ones((s, s), dtype=jnp.bool_), k=1)
        scores = jnp.where(mask[None, None], -1e9, scores)
        attn = jax.nn.softmax(scores, axis=-1)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, -1)
        h = ctx @ blk["o"].T
        x = h + residual

        residual = x
        h = _rms_norm(x, blk["ffn_norm"], cfg.rms_norm_eps)
        h = (jax.nn.silu(h @ blk["gate"].T) * (h @ blk["up"].T)) @ blk["down"].T
        x = h + residual

    x = _rms_norm(x, params["final_norm"], cfg.rms_norm_eps)
    return x @ params["lm_head"].T


def causal_lm_loss(params, tokens, cfg, cos, sin):
    logits = forward(params, tokens, cfg, cos, sin)
    shift_logits = logits[:, :-1, :]
    shift_labels = tokens[:, 1:]
    logp = jax.nn.log_softmax(shift_logits, axis=-1)
    nll = -jnp.take_along_axis(logp, shift_labels[..., None], axis=-1)[..., 0]
    return nll.mean()


# ---------------------------------------------------------------------------
# AdamW (from scratch, JAX)
# ---------------------------------------------------------------------------
def adamw_init(params):
    return {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
        "t": 0,
    }


def adamw_update(params, grads, state, lr, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, state["v"], grads)
    bc1 = 1 - b1 ** t
    bc2 = 1 - b2 ** t
    params = jax.tree_util.tree_map(
        lambda p, m_, v_: p - lr * (wd * p + (m_ / bc1) / (jnp.sqrt(v_ / bc2) + eps)),
        params, m, v)
    return params, {"m": m, "v": v, "t": t}


def _global_norm(grads):
    return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))


def _clip_grads(grads, clip):
    gnorm = _global_norm(grads)
    scale = jnp.minimum(1.0, clip / (gnorm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


# ---------------------------------------------------------------------------
# data loading (reads the same shards as the PyTorch path)
# ---------------------------------------------------------------------------
def load_tokens(data_dir: str) -> np.ndarray:
    manifest_path = os.path.join(data_dir, "shards_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    dtype = np.uint16 if manifest["dtype"] == "uint16" else np.uint32
    arrays = []
    for shard in manifest["shards"]:
        path = os.path.join(data_dir, shard["file"])
        arrays.append(np.fromfile(path, dtype=dtype))
    return np.concatenate(arrays).astype(np.int32)


def batch_iterator(tokens: np.ndarray, batch_size: int, seq_len: int, seed: int = 0):
    """Infinite generator of (batch_size, seq_len) token batches."""
    rng = np.random.default_rng(seed)
    n = tokens.shape[0]
    assert n > seq_len, f"corpus too small ({n} tokens) for seq_len {seq_len}"
    while True:
        starts = rng.integers(0, n - seq_len, size=batch_size)
        batch = np.stack([tokens[s:s + seq_len] for s in starts])
        yield jnp.asarray(batch, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------
def save_params(params: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    leaves, _ = jax.tree_util.tree_flatten(params)
    np.savez(path, **{f"arr_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)})


def load_params(path: str, template: Dict) -> Dict:
    data = np.load(path)
    leaves = [data[f"arr_{i}"] for i in range(len(data.files))]
    _, treedef = jax.tree_util.tree_flatten(template)
    return jax.tree_util.tree_unflatten(treedef, leaves)


# ---------------------------------------------------------------------------
# training (data-parallel across TPU cores via jax.pmap)
# ---------------------------------------------------------------------------
def train_on_tpu(cfg: JaxModelConfig, tokens: np.ndarray, total_steps: int,
                 batch_size_per_device: int, seq_len: int, lr: float = 3e-4,
                 min_lr: float = 3e-5, warmup: int = 100, grad_clip: float = 1.0,
                 weight_decay: float = 0.1, ckpt_dir: str = "checkpoints_tpu",
                 log_every: int = 10, save_every: int = 1000,
                 seed: int = 0, use_pmap: bool = True):
    if not _HAS_JAX:
        raise RuntimeError("JAX is not installed; run this on a TPU VM.")

    devices = jax.devices()
    n_devices = len(devices)
    print(f"[MC-LLM JAX] {n_devices} device(s): {[str(d) for d in devices]}")

    key = random.PRNGKey(seed)
    params = init_params(cfg, key)
    cos, sin = _precompute_rope(cfg)
    state = adamw_init(params)

    def lr_at(step):
        if step < warmup:
            return lr * (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    it = batch_iterator(tokens, batch_size_per_device * n_devices, seq_len, seed)

    def make_step(avg_grad: bool):
        def step(p, s, t, lr_val):
            loss, grads = jax.value_and_grad(causal_lm_loss, argnums=0)(p, t, cfg, cos, sin)
            if avg_grad:
                loss = jax.lax.pmean(loss, axis_name="data")
                grads = jax.tree_util.tree_map(
                    lambda g: jax.lax.pmean(g, axis_name="data"), grads)
            grads = _clip_grads(grads, grad_clip)
            p, s = adamw_update(p, grads, s, lr_val, wd=weight_decay)
            return loss, p, s
        return step

    if use_pmap and n_devices > 1:
        step_fn = jax.pmap(make_step(True), axis_name="data")
        params = jax.device_put_replicated(params, devices)
        state = jax.device_put_replicated(state, devices)
    else:
        step_fn = jax.jit(make_step(False))

    t0 = time.time()
    for step in range(total_steps):
        batch = next(it)
        if use_pmap and n_devices > 1:
            batch = batch.reshape(n_devices, batch_size_per_device, seq_len)
            batch = jax.device_put_sharded(list(batch), devices)
        loss, params, state = step_fn(params, state, batch, lr_at(step))

        if use_pmap and n_devices > 1:
            loss_val = float(jax.device_get(loss)[0])
        else:
            loss_val = float(loss)

        if step % log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_sec = batch_size_per_device * n_devices * seq_len / max(dt, 1e-6)
            print(f"step {step:>6}/{total_steps} loss {loss_val:.4f} "
                  f"lr {lr_at(step):.2e} {tok_sec:.0f} tok/s")

        if step % save_every == 0 and step > 0:
            p_cpu = jax.device_get(params)
            if use_pmap and n_devices > 1:
                p_cpu = jax.tree_util.tree_map(lambda x: x[0], p_cpu)
            save_params(p_cpu, os.path.join(ckpt_dir, f"step_{step}.npz"))
            print(f"[MC-LLM JAX] saved checkpoint step {step}")

    p_cpu = jax.device_get(params)
    if use_pmap and n_devices > 1:
        p_cpu = jax.tree_util.tree_map(lambda x: x[0], p_cpu)
    save_params(p_cpu, os.path.join(ckpt_dir, "final.npz"))
    return p_cpu


@jax.jit
def _greedy_step(params, tokens, cfg, cos, sin):
    logits = forward(params, tokens, cfg, cos, sin)
    return logits[:, -1, :].argmax(-1)


def greedy_generate(params, prompt_ids, cfg, max_new_tokens: int) -> jnp.ndarray:
    cos, sin = _precompute_rope(cfg)
    tokens = jnp.asarray([prompt_ids], dtype=jnp.int32)
    for _ in range(max_new_tokens):
        nxt = _greedy_step(params, tokens[:, -cfg.max_seq_len:], cfg, cos, sin)
        tokens = jnp.concatenate([tokens, nxt[:, None]], axis=1)
    return tokens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train MC-LLM on TPU (JAX)")
    parser.add_argument("--config", default="configs/100m.yaml")
    parser.add_argument("--data-dir", default="data/shards")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=8, help="batch per device")
    parser.add_argument("--seq-len", type=int, default=0,
                        help="sequence length (0 = use config's max_position_embeddings)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--out", default="checkpoints_tpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-pmap", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="random-token smoke test (no data needed)")
    args = parser.parse_args()

    if not _HAS_JAX:
        raise RuntimeError("JAX is not installed. Run on a TPU VM with "
                           "`pip install \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`")

    if os.path.exists(args.config):
        cfg = JaxModelConfig.from_yaml(args.config)
    else:
        cfg = JaxModelConfig()
    if args.seq_len:
        cfg.max_seq_len = args.seq_len

    print("=" * 52)
    print("MEGA-CYBER LLM (JAX/TPU)")
    print(f"parameters: {cfg.total_parameters():,}")
    print(f"hidden {cfg.hidden_size}, layers {cfg.num_layers}, "
          f"heads {cfg.num_heads}, vocab {cfg.vocab_size}")
    print("=" * 52)

    if args.smoke:
        tokens = np.random.randint(0, cfg.vocab_size, size=1_000_000).astype(np.int32)
        print("[smoke] using random tokens (no real data)")
    else:
        tokens = load_tokens(args.data_dir)
        print(f"[data] loaded {tokens.shape[0]:,} tokens from {args.data_dir}")

    train_on_tpu(
        cfg, tokens, total_steps=args.steps,
        batch_size_per_device=args.batch, seq_len=cfg.max_seq_len,
        lr=args.lr, warmup=args.warmup, ckpt_dir=args.out,
        seed=args.seed, use_pmap=not args.no_pmap,
    )


if __name__ == "__main__":
    main()
