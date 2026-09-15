"""Resource monitoring for Colab dataset generation.

Tracks RAM, free disk, CPU, elapsed time, dataset size and token count, and
exposes safe stop/flush thresholds.  All limits are configurable so the
generator can use as much of the free Colab runtime as possible without
crashing it.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass
class ResourceLimits:
    ram_fraction: float = 0.85        # stop/flush above this RAM fraction
    disk_limit_gb: float = 8.0        # stop when free disk drops below this
    max_runtime_seconds: float = 0.0  # 0 = unlimited
    max_tokens: int = 0               # 0 = unlimited
    max_dialogues: int = 0            # 0 = unlimited
    max_shards: int = 0               # 0 = unlimited


@dataclass
class ResourceSnapshot:
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_fraction: float = 0.0
    disk_free_gb: float = 0.0
    cpu_percent: float = 0.0
    elapsed_seconds: float = 0.0


class ResourceMonitor:
    """Polls system resources and decides when to flush / stop."""

    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.start_time = time.time()
        self.peak_ram_gb = 0.0
        self.peak_disk_gb = 0.0
        self._proc = psutil.Process() if _HAS_PSUTIL else None

    def snapshot(self) -> ResourceSnapshot:
        snap = ResourceSnapshot(elapsed_seconds=time.time() - self.start_time)
        if _HAS_PSUTIL:
            vm = psutil.virtual_memory()
            snap.ram_total_gb = vm.total / 1e9
            snap.ram_used_gb = (vm.total - vm.available) / 1e9
            snap.ram_fraction = snap.ram_used_gb / max(1e-9, snap.ram_total_gb)
            snap.cpu_percent = psutil.cpu_percent(interval=None)
        try:
            _, _, free = shutil.disk_usage(os.getcwd())
            snap.disk_free_gb = free / 1e9
        except Exception:
            snap.disk_free_gb = 1e9
        self.peak_ram_gb = max(self.peak_ram_gb, snap.ram_used_gb)
        self.peak_disk_gb = max(self.peak_disk_gb, snap.ram_used_gb)
        return snap

    def should_flush(self, snap: ResourceSnapshot) -> bool:
        return snap.ram_fraction >= self.limits.ram_fraction

    def should_stop(self, snap: ResourceSnapshot,
                    dialogues: int = 0, tokens: int = 0,
                    shards: int = 0) -> tuple:
        """Return (stop, reason)."""
        L = self.limits
        if snap.ram_fraction >= 0.95:
            return True, "RAM > 95% (crash risk)"
        if L.disk_limit_gb > 0 and snap.disk_free_gb < L.disk_limit_gb:
            return True, f"free disk {snap.disk_free_gb:.1f} GB < limit {L.disk_limit_gb} GB"
        if L.max_runtime_seconds > 0 and snap.elapsed_seconds >= L.max_runtime_seconds:
            return True, f"runtime limit {L.max_runtime_seconds}s reached"
        if L.max_tokens > 0 and tokens >= L.max_tokens:
            return True, "token limit reached"
        if L.max_dialogues > 0 and dialogues >= L.max_dialogues:
            return True, "dialogue limit reached"
        if L.max_shards > 0 and shards >= L.max_shards:
            return True, "shard limit reached"
        return False, ""


def format_snapshot(snap: ResourceSnapshot) -> str:
    return (f"RAM {snap.ram_fraction:.0%} ({snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f} GB) | "
            f"disk {snap.disk_free_gb:.0f} GB free | cpu {snap.cpu_percent:.0f}% | "
            f"t {snap.elapsed_seconds/60:.1f} min")
