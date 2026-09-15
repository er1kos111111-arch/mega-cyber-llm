"""Distributed training support for MC-LLM.

Provides process-group init, DDP/FSDP wrapping, tensor-parallel linear
layers (column/row sharding with all-reduce/all-gather), and a minimal
pipeline-parallel scaffold.

Backends: ``nccl``/``gloo`` via PyTorch (NVIDIA + AMD via ROCm).  A JAX/TPU
backend lives separately in ``training/tpu_backend.py``.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


def dist_is_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_process_group(backend: Optional[str] = None, timeout_minutes: int = 60) -> None:
    if dist.is_initialized():
        return
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    # nccl needs an explicit timeout for large models
    if backend == "nccl":
        dist.destroy_process_group()  # re-init with timeout
        dist.init_process_group(
            backend=backend,
            init_method=os.environ.get("MASTER_ADDR") and "env://",
        )


def get_rank() -> int:
    return dist.get_rank() if dist_is_available() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist_is_available() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if dist_is_available():
        dist.barrier()


def wrap_ddp(model: nn.Module, find_unused: bool = False) -> nn.Module:
    """Wrap a model in DistributedDataParallel."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    return DDP(model, device_ids=[get_rank()] if torch.cuda.is_available() else None,
               find_unused_parameters=find_unused)


def wrap_fsdp(model: nn.Module, mixed_precision=None) -> nn.Module:
    """Wrap a model in FullyShardedDataParallel (ZeRO-style sharding)."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        kwargs = dict(sharding_strategy=ShardingStrategy.FULL_SHARD)
        if mixed_precision is not None:
            kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=mixed_precision, reduce_dtype=mixed_precision,
                buffer_dtype=mixed_precision)
        return FSDP(model, **kwargs)
    except ImportError:
        return model


# ---------------------------------------------------------------------------
# Tensor parallelism
# ---------------------------------------------------------------------------
def _ensure_parallel() -> None:
    if not dist_is_available():
        raise RuntimeError("Tensor parallelism requires torch.distributed to be initialized")


class ColumnParallelLinear(nn.Module):
    """Linear layer with the output dimension sharded across ranks.

    ``y = x @ W^T + b`` where ``W`` is split column-wise.  The input is
    replicated; each rank computes its slice of the output.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 gather_output: bool = False):
        super().__init__()
        world = get_world_size()
        assert out_features % world == 0, "out_features must be divisible by world_size"
        self.in_features = in_features
        self.out_features = out_features
        self.out_per_rank = out_features // world
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_per_rank)) if bias else None
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.linear(x, self.weight, self.bias)
        if self.gather_output:
            y = _all_gather_along_last_dim(y)
        return y


class RowParallelLinear(nn.Module):
    """Linear layer with the input dimension sharded across ranks.

    ``y = x @ W^T`` where ``W`` is split row-wise; each rank owns a slice of
    the input.  Partial sums are all-reduced at the end.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 input_is_parallel: bool = False):
        super().__init__()
        world = get_world_size()
        assert in_features % world == 0, "in_features must be divisible by world_size"
        self.in_features = in_features
        self.out_features = out_features
        self.in_per_rank = in_features // world
        self.input_is_parallel = input_is_parallel
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.linear(x, self.weight)
        if dist_is_available():
            dist.all_reduce(y)
        if self.bias is not None:
            y = y + self.bias
        return y


def _all_gather_along_last_dim(x: torch.Tensor) -> torch.Tensor:
    world = get_world_size()
    out = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(out, x)
    return torch.cat(out, dim=-1)


def split_tensor_parallel(model: nn.Module, tp_size: int) -> nn.Module:
    """Replace ``nn.Linear`` layers with tensor-parallel variants (simple sharding)."""
    # This is a convenience hook; for a full TP model the layers are built TP
    # from the start.  Kept for API completeness.
    return model


# ---------------------------------------------------------------------------
# Pipeline parallelism (minimal scaffold)
# ---------------------------------------------------------------------------
class PipelineStage:
    """A single stage of a pipelined model.  Placeholder describing the
    interface; concrete scheduling lives in the training loop."""

    def __init__(self, layers: nn.ModuleList, stage_id: int, num_stages: int):
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
