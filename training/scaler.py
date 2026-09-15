"""Mixed-precision scaler for MC-LLM training.

Wraps ``torch.cuda.amp.GradScaler`` when FP16 is requested and a CUDA
backend is available; otherwise acts as a no-op (float32 training).  BF16
training is handled separately via ``torch.autocast`` in the training loop.
"""
from __future__ import annotations

import torch


def supports_bf16(device: str = "cuda") -> bool:
    if device == "cuda":
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if device == "cpu":
        try:
            return torch.cpu.is_bf16_supported() if hasattr(torch, "cpu") else False
        except Exception:
            return False
    return False


def build_scaler(dtype: str, enabled: bool = True):
    """Return a ``torch.cuda.amp.GradScaler`` or ``None``."""
    if dtype == "float16" and enabled and torch.cuda.is_available():
        return torch.cuda.amp.GradScaler(enabled=True)
    return None


def resolve_dtype(dtype: str, device: str = "cuda") -> str:
    """Resolve a requested dtype to one the device actually supports.

    e.g. a Tesla T4 does not accelerate BF16, so we fall back to FP16.
    """
    dtype = (dtype or "float32").lower()
    if dtype not in ("bfloat16", "float16"):
        return "float32"
    if device == "cuda" and torch.cuda.is_available():
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            return "float16"
        return dtype
    return "float32"


def autocast_context(dtype: str, device: str = "cuda"):
    """Return an autocast context manager for the given dtype, or a nullcontext."""
    resolved = resolve_dtype(dtype, device)
    if resolved in ("bfloat16", "float16") and torch.cuda.is_available():
        amp_dtype = torch.bfloat16 if resolved == "bfloat16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
