"""Dataset sharding for the MC-LLM data pipeline.

Converts tokenized documents into binary shards for memory-mapped
streaming.  Each shard is a flat array of token ids (``uint16``/``uint32``
depending on vocabulary size) with document boundaries recorded in a
JSON manifest, so the training dataloader never needs the whole dataset in
RAM.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional

import numpy as np

_DTYPE = {16: np.uint16, 32: np.uint32}


def pick_dtype(vocab_size: int):
    return _DTYPE[16] if vocab_size <= 65535 else _DTYPE[32]


def read_texts(path: str) -> Iterator[str]:
    """Stream documents from a supported file type (JSONL / TXT / MD / HTML).

    JSONL lines are expected to be ``{"text": "..."}`` or a bare string.
    TXT/MD/HTML are read line-by-line with paragraph grouping.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, str):
                    yield obj
                elif isinstance(obj, dict) and "text" in obj:
                    yield obj["text"]
    else:
        buf = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip() == "":
                    if buf:
                        yield "\n".join(buf)
                        buf = []
                else:
                    buf.append(line)
            if buf:
                yield "\n".join(buf)


class ShardWriter:
    """Tokenizes documents and writes them to binary shards."""

    def __init__(self, out_dir: str, vocab_size: int, tokens_per_shard: int = 1_000_000,
                 max_seq_len: Optional[int] = None):
        self.out_dir = out_dir
        self.dtype = pick_dtype(vocab_size)
        self.tokens_per_shard = tokens_per_shard
        self.max_seq_len = max_seq_len
        self.shards: List[Dict] = []
        self._buf: List[int] = []
        self._shard_idx = 0
        os.makedirs(out_dir, exist_ok=True)

    def add_document(self, token_ids: List[int]) -> None:
        if not token_ids:
            return
        if self.max_seq_len is not None:
            for i in range(0, len(token_ids), self.max_seq_len):
                chunk = token_ids[i:i + self.max_seq_len]
                if chunk:
                    self._buf.extend(chunk)
                    self._flush_if_needed()
        else:
            self._buf.extend(token_ids)
            self._flush_if_needed()

    def _flush_if_needed(self) -> None:
        while len(self._buf) >= self.tokens_per_shard:
            chunk = self._buf[:self.tokens_per_shard]
            self._buf = self._buf[self.tokens_per_shard:]
            self._write_shard(chunk)

    def _write_shard(self, tokens: List[int]) -> None:
        arr = np.asarray(tokens, dtype=self.dtype)
        name = f"shard_{self._shard_idx:05d}.bin"
        path = os.path.join(self.out_dir, name)
        arr.tofile(path)
        self.shards.append({"file": name, "tokens": int(arr.size)})
        self._shard_idx += 1

    def finish(self) -> str:
        if self._buf:
            self._write_shard(self._buf)
            self._buf = []
        manifest = {
            "dtype": "uint16" if self.dtype == np.uint16 else "uint32",
            "num_shards": len(self.shards),
            "total_tokens": sum(s["tokens"] for s in self.shards),
            "shards": self.shards,
        }
        manifest_path = os.path.join(self.out_dir, "shards_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path


def shard_texts(texts: Iterator[str], tokenizer, out_dir: str, vocab_size: int,
                tokens_per_shard: int = 1_000_000, max_seq_len: int = 8192,
                add_eos: bool = True) -> str:
    """Tokenize an iterator of documents and write shards."""
    writer = ShardWriter(out_dir, vocab_size, tokens_per_shard, max_seq_len)
    for text in texts:
        ids = tokenizer.encode(text)
        if add_eos:
            ids = ids + [tokenizer.eos_token_id]
        writer.add_document(ids)
    return writer.finish()
