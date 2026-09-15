"""Deduplication for generated conversations.

Three levels, per spec:

* **exact** — identical dialogues (Bloom filter + session hash set);
* **near** — dialogues differing by a few words (MinHash / LSH);
* **template** — the same template sequence with swapped slot values.

Everything is memory-bounded so it can run for a long time in Colab without
growing without limit.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Set, Tuple

from data.dedup import MinHashDedup


def _normalize_exact(text: str) -> str:
    return " ".join(text.split())


class BloomFilter:
    """Space-efficient exact-duplicate detector."""

    def __init__(self, capacity: int = 50_000_000, error_rate: float = 0.001):
        self.size = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.num_hashes = max(1, int(self.size / capacity * math.log(2)))
        self.bits = bytearray((self.size + 7) // 8)

    def _indices(self, s: str):
        h1 = int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest(), "big")
        h2 = int.from_bytes(hashlib.md5(s.encode("utf-8")).digest(), "big")
        for i in range(self.num_hashes):
            yield (h1 + i * h2 + i * i) % self.size

    def add(self, s: str) -> None:
        for h in self._indices(s):
            self.bits[h // 8] |= 1 << (h % 8)

    def __contains__(self, s: str) -> bool:
        return all(self.bits[h // 8] & (1 << (h % 8)) for h in self._indices(s))


class ConversationDedup:
    def __init__(self, bloom_capacity: int = 50_000_000,
                 template_max_repeat: int = 3,
                 near_threshold: float = 0.92,
                 near_max_store: int = 100_000,
                 template_max_store: int = 2_000_000):
        self.bloom = BloomFilter(bloom_capacity)
        self.exact_seen: Set[str] = set()
        self.template_counts: Dict[Tuple, int] = {}
        self.template_max_repeat = template_max_repeat
        self.template_max_store = template_max_store
        self.near = MinHashDedup(threshold=near_threshold)
        self.near_count = 0
        self.near_max_store = near_max_store
        self.stats = {"exact": 0, "near": 0, "template": 0}

    def should_keep(self, dialogue: Dict) -> Tuple[bool, str]:
        text = dialogue["text"]

        # 1. exact
        key = _normalize_exact(text)
        if key in self.bloom or key in self.exact_seen:
            self.stats["exact"] += 1
            return False, "exact"

        # 2. template (signature is the sequence of templates, no slot values)
        sig = dialogue.get("signature")
        if sig is not None:
            c = self.template_counts.get(sig, 0)
            if c >= self.template_max_repeat:
                self.stats["template"] += 1
                return False, "template"

        # 3. near
        if self.near_count < self.near_max_store and self.near.is_duplicate(text):
            self.stats["near"] += 1
            return False, "near"

        return True, "ok"

    def record(self, dialogue: Dict) -> None:
        key = _normalize_exact(dialogue["text"])
        self.bloom.add(key)
        self.exact_seen.add(key)
        sig = dialogue.get("signature")
        if sig is not None:
            self.template_counts[sig] = self.template_counts.get(sig, 0) + 1
        self.near_count += 1
        # bound long-term memory
        if len(self.template_counts) > self.template_max_store:
            self.template_counts.clear()

    def release_session_memory(self) -> None:
        """Drop the exact-seen set (Bloom filter retains long-term memory)."""
        self.exact_seen.clear()
        self.near_count = 0
