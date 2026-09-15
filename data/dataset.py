"""Streaming dataset for MC-LLM pretraining.

A PyTorch ``IterableDataset`` that memory-maps the binary shards produced
by :mod:`data.shard` and yields fixed-length token sequences.  It never
loads the full dataset into RAM — only the currently-mapped shard is
resident, and shards are opened lazily.
"""
from __future__ import annotations

import json
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


class MemmapShard:
    def __init__(self, path: str, dtype):
        self.arr = np.memmap(path, dtype=dtype, mode="r")
        self.tokens = int(self.arr.size)

    def close(self) -> None:
        del self.arr


class ShardedTokenDataset(IterableDataset):
    """Streams sequences of length ``seq_len`` from binary shards.

    Supports both contiguous packing and randomly-sampled sequences.  Uses
    a sliding window with a configurable stride to maximize token coverage.
    """

    def __init__(self, data_dir: str, seq_len: int, stride: Optional[int] = None,
                 seed: int = 0, shuffle_shards: bool = True,
                 num_shards_per_sample: Optional[int] = None):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.seed = seed
        self.shuffle_shards = shuffle_shards

        manifest_path = os.path.join(data_dir, "shards_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.dtype = np.uint16 if manifest["dtype"] == "uint16" else np.uint32
        self.shards = manifest["shards"]
        self.total_tokens = manifest["total_tokens"]

    def _open_shard(self, shard) -> MemmapShard:
        path = os.path.join(self.data_dir, shard["file"])
        return MemmapShard(path, self.dtype)

    def _sample_sequences(self, arr, max_seq_len: int) -> Iterator[torch.Tensor]:
        # contiguous sliding window
        n = arr.size
        for start in range(0, max(0, n - self.seq_len + 1), self.stride):
            seq = arr[start:start + self.seq_len]
            if seq.size < self.seq_len:
                break
            yield torch.from_numpy(np.asarray(seq, dtype=np.int64))

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        order = list(range(len(self.shards)))
        if self.shuffle_shards:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(order)
        if worker_info is not None:
            # each worker processes a disjoint subset of shards
            order = order[worker_info.id:: worker_info.num_workers]
        for idx in order:
            shard = self._open_shard(self.shards[idx])
            try:
                yield from self._sample_sequences(shard.arr, shard.tokens)
            finally:
                shard.close()


class StreamingChatDataset(IterableDataset):
    """Yields (tokens, labels) pairs for causal LM from a token stream.

    ``input`` = tokens[0:n-1], ``target`` = tokens[1:n] (see training loop).
    """

    def __init__(self, data_dir: str, seq_len: int, **kwargs):
        self.inner = ShardedTokenDataset(data_dir, seq_len, **kwargs)

    def __iter__(self):
        for seq in self.inner:
            yield seq[:-1], seq[1:]


def build_dataloader(data_dir: str, seq_len: int, batch_size: int,
                     num_workers: int = 0, seed: int = 0,
                     drop_last: bool = True) -> torch.utils.data.DataLoader:
    dataset = StreamingChatDataset(data_dir, seq_len, seed=seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        drop_last=drop_last, pin_memory=False,
    )
