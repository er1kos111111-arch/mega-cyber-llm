"""System information + automatic config recommendation for MC-LLM.

Detects CPU, RAM, GPU/VRAM, CUDA/ROCm/TPU, BF16/FP8 support, and disk, then
recommends the largest config that fits on the current hardware.
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import Optional


def _cpu_info() -> str:
    import platform
    return platform.processor() or platform.machine()


def _ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 0.0


def _gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            vram = []
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                vram.append(props.total_memory / (1024 ** 3))
            return {
                "backend": "CUDA", "count": count, "vram_gb": vram,
                "bf16": torch.cuda.is_bf16_supported(),
            }
    except Exception:
        pass

    # ROCm / AMD
    try:
        import torch
        if hasattr(torch, "hip") and torch.hip.is_available():
            return {"backend": "ROCm", "count": torch.hip.device_count(),
                    "vram_gb": [], "bf16": False}
    except Exception:
        pass
    return {"backend": None, "count": 0, "vram_gb": [], "bf16": False}


def _tpu_info() -> Optional[str]:
    try:
        import jax
        devices = jax.devices()
        tpu = [d for d in devices if "TPU" in str(d).upper()]
        if tpu:
            return f"{len(tpu)} TPU devices (type {tpu[0].device_kind})"
    except Exception:
        pass
    return None


def _disk_gb() -> float:
    try:
        total, used, free = shutil.disk_usage(os.getcwd())
        return free / (1024 ** 3)
    except Exception:
        return 0.0


def detect() -> dict:
    gpu = _gpu_info()
    return {
        "python": sys.version.split()[0],
        "cpu": _cpu_info(),
        "ram_gb": round(_ram_gb(), 1),
        "gpu": gpu,
        "tpu": _tpu_info(),
        "disk_free_gb": round(_disk_gb(), 1),
    }


# approximate minimum VRAM (GB) for each config (bf16 weights + optim + grad)
CONFIG_VRAM = {
    "100m": 3, "300m": 6, "1b": 18, "3b": 45, "7b": 90,
    "13b": 160, "70b": 800, "405b": 4600, "800b": 9000,
}


def recommend(info: dict) -> str:
    gpu = info["gpu"]
    if gpu["count"] > 0:
        total_vram = sum(gpu["vram_gb"]) if gpu["vram_gb"] else gpu["count"] * 80
        for name in ["800b", "405b", "70b", "13b", "7b", "3b", "1b", "300m", "100m"]:
            if total_vram >= CONFIG_VRAM[name]:
                return f"{name} (total VRAM {total_vram:.0f} GB)"
        return "100m"
    if info["tpu"]:
        return "70b+ (TPU pod; use the JAX backend + tensor/pipeline parallelism)"
    ram = info["ram_gb"]
    if ram >= 64:
        return "1b (CPU/limited, will be slow)"
    if ram >= 16:
        return "100m (CPU)"
    return "tiny smoke config (CPU)"


def main():
    info = detect()
    rec = recommend(info)

    print("=" * 60)
    print("MEGA-CYBER LLM — system detection")
    print("=" * 60)
    print(f"CPU:              {info['cpu']}")
    print(f"RAM:              {info['ram_gb']} GB")
    g = info["gpu"]
    if g["count"] > 0:
        print(f"GPU backend:      {g['backend']}")
        print(f"GPUs:             {g['count']}")
        print(f"VRAM each:        {g['vram_gb']}")
        print(f"BF16 support:     {g['bf16']}")
    else:
        print("GPU:              none")
    print(f"TPU:              {info['tpu'] or 'none'}")
    print(f"Free disk:        {info['disk_free_gb']} GB")
    print("-" * 60)
    print(f"Recommended:      {rec}")
    print("=" * 60)


if __name__ == "__main__":
    main()
