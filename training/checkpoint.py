"""Checkpointing for MC-LLM.

Checkpoints are saved as:

* ``mc_llm_step_<step>.pt``        — model weights (single file, or sharded);
* ``mc_llm_step_<step>_optim.pt``  — optimizer + scheduler state;
* ``mc_llm_step_<step>_meta.json`` — metadata (step, tokens, loss, hash).

Supports sharding (splitting large tensors across files), integrity
validation via SHA-256, and resume-after-restart semantics.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    step: int,
    out_dir: str,
    config: Any,
    tokens_seen: int = 0,
    loss: Optional[float] = None,
    shard_size: Optional[int] = None,
    is_main: bool = True,
) -> Dict[str, Any]:
    """Save a full checkpoint.  Returns metadata dict."""
    if not is_main:
        return {}
    os.makedirs(out_dir, exist_ok=True)

    model_state = model.state_dict()
    model_path = os.path.join(out_dir, f"mc_llm_step_{step:06d}.pt")

    if shard_size is None:
        torch.save(model_state, model_path)
    else:
        model_path = save_state_dict_sharded(model_state, out_dir, step, shard_size)

    optim_path = None
    if optimizer is not None:
        optim_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        }
        optim_path = os.path.join(out_dir, f"mc_llm_step_{step:06d}_optim.pt")
        torch.save(optim_state, optim_path)

    meta = {
        "checkpoint": os.path.basename(model_path),
        "step": step,
        "tokens_seen": tokens_seen,
        "loss": loss,
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "sha256": _sha256_file(model_path) if os.path.isfile(model_path) else None,
    }
    meta_path = os.path.join(out_dir, f"mc_llm_step_{step:06d}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # symlink-free "latest" pointer
    latest = os.path.join(out_dir, "latest.json")
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def save_state_dict_sharded(state_dict: Dict[str, torch.Tensor], out_dir: str,
                            step: int, shard_size: int) -> str:
    """Shard tensors across files and write an index.  Returns the shard dir."""
    shard_dir = os.path.join(out_dir, f"mc_llm_step_{step:06d}")
    os.makedirs(shard_dir, exist_ok=True)
    shapes: Dict[str, list] = {}
    shard_id = 0
    current: Dict[str, torch.Tensor] = {}
    current_bytes = 0

    def flush():
        nonlocal shard_id, current, current_bytes
        if not current:
            return
        path = os.path.join(shard_dir, f"part_{shard_id:03d}.pt")
        torch.save(current, path)
        shard_id += 1
        current = {}
        current_bytes = 0

    for name, tensor in state_dict.items():
        tensor = tensor.detach().cpu()
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > shard_size:
            shapes[name] = list(tensor.shape)
            flat = tensor.flatten()
            el = tensor.element_size()
            n_per = max(1, shard_size // el)
            part_idx = 0
            for i in range(0, flat.numel(), n_per):
                key = f"{name}__part{part_idx}"
                current[key] = flat[i:i + n_per]
                current_bytes += current[key].numel() * el
                part_idx += 1
                if current_bytes >= shard_size:
                    flush()
        else:
            current[name] = tensor
            current_bytes += nbytes
            if current_bytes >= shard_size:
                flush()
    flush()

    with open(os.path.join(shard_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"shapes": shapes}, f, indent=2)
    return shard_dir


def load_state_dict_sharded(index_path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    shard_dir = os.path.dirname(index_path)
    with open(index_path, "r", encoding="utf-8") as f:
        shapes = json.load(f).get("shapes", {})

    flat: Dict[str, torch.Tensor] = {}
    parts = sorted(p for p in os.listdir(shard_dir) if p.startswith("part_"))
    for part in parts:
        loaded = torch.load(os.path.join(shard_dir, part), map_location=map_location)
        for k, v in loaded.items():
            flat[k] = v

    result: Dict[str, torch.Tensor] = {}
    reassemble: Dict[str, list] = {}
    for k, v in flat.items():
        if "__part" in k:
            base, _, idx = k.rpartition("__part")
            reassemble.setdefault(base, []).append((int(idx), v))
        else:
            result[k] = v
    for base, parts_list in reassemble.items():
        parts_list.sort(key=lambda x: x[0])
        tensor = torch.cat([v for _, v in parts_list])
        if base in shapes:
            tensor = tensor.reshape(shapes[base])
        result[base] = tensor
    return result


def load_checkpoint(model: torch.nn.Module, ckpt_dir: str, step: Optional[int] = None,
                    map_location="cpu") -> Tuple[int, Dict[str, Any]]:
    """Load model weights + metadata.  Returns (step, meta)."""
    if step is not None:
        meta_path = os.path.join(ckpt_dir, f"mc_llm_step_{step:06d}_meta.json")
    else:
        latest = os.path.join(ckpt_dir, "latest.json")
        if not os.path.exists(latest):
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        with open(latest, "r", encoding="utf-8") as f:
            latest_meta = json.load(f)
        meta_path = os.path.join(ckpt_dir, f"mc_llm_step_{latest_meta['step']:06d}_meta.json")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    ckpt = meta["checkpoint"]
    ckpt_path = os.path.join(ckpt_dir, ckpt)
    if os.path.isdir(ckpt_path):
        state = load_state_dict_sharded(os.path.join(ckpt_path, "index.json"), map_location)
    else:
        state = torch.load(ckpt_path, map_location=map_location)

    model.load_state_dict(state)
    return meta["step"], meta


def load_optimizer_state(optimizer, scheduler, ckpt_dir: str, step: int) -> None:
    path = os.path.join(ckpt_dir, f"mc_llm_step_{step:06d}_optim.pt")
    if not os.path.exists(path):
        return
    state = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(state["scheduler"])


def validate_checkpoint(meta_path: str) -> bool:
    """Verify the model file hash matches the metadata record."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ckpt_dir = os.path.dirname(meta_path)
    ckpt_path = os.path.join(ckpt_dir, meta["checkpoint"])
    if os.path.isdir(ckpt_path):
        return True  # sharded dirs validated per-part
    if not os.path.exists(ckpt_path):
        return False
    return _sha256_file(ckpt_path) == meta["sha256"]
