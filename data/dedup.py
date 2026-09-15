"""Deduplication for the MC-LLM data pipeline.

Two strategies:

* ``exact`` — SHA-256 hashing of normalized documents (streaming, memory
  bounded by the number of unique hashes);
* ``minhash`` — locality-sensitive hashing to detect near-duplicates using
  a fixed set of 128 permutations over character/word n-grams.

Both are implemented from scratch with no external dedup service.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable, Iterator, Optional, Set


def _norm(doc: str) -> str:
    return " ".join(doc.split())


def doc_hash(doc: str) -> str:
    return hashlib.sha256(_norm(doc).encode("utf-8")).hexdigest()


def exact_dedup(docs: Iterable[str]) -> Iterator[str]:
    """Yield only documents whose normalized hash has not been seen before."""
    seen: Set[str] = set()
    for doc in docs:
        h = doc_hash(doc)
        if h in seen:
            continue
        seen.add(h)
        yield doc


# ---------------------------------------------------------------------------
# MinHash (near-duplicate) detection
# ---------------------------------------------------------------------------
_NUM_PERM = 128
_MASK32 = (1 << 32) - 1
_MERSENNE = (1 << 61) - 1


def _shingles(text: str, n: int = 5) -> Set[int]:
    """Return hashed character n-grams (a.k.a. shingles)."""
    text = _norm(text)
    out: Set[int] = set()
    for i in range(max(1, len(text) - n + 1)):
        gram = text[i:i + n]
        out.add(hashlib.md5(gram.encode("utf-8")).digest().__hash__())
    return out


class MinHashDedup:
    """Streaming near-duplicate filter based on MinHash signatures."""

    def __init__(self, num_perm: int = _NUM_PERM, threshold: float = 0.8,
                 shingle_size: int = 5):
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        # two random linear hash functions -> 128 permutations via a*X+b
        self._a = [self._rand() for _ in range(num_perm)]
        self._b = [self._rand() for _ in range(num_perm)]
        # bucket signatures -> doc ids (LSH banding)
        self._bands: dict = defaultdict(list)
        self._band_size = 4
        self._num_bands = num_perm // self._band_size
        self._signatures: dict = {}
        self._next_id = 0

    def _rand(self) -> int:
        import random
        return random.randint(1, _MERSENNE - 1)

    def _minhash(self, shingles: Set[int]) -> list:
        sig = []
        for a, b in zip(self._a, self._b):
            m = _MERSENNE
            for x in shingles:
                h = (a * x + b) & _MASK32
                if h < m:
                    m = h
            sig.append(m)
        return sig

    def _jaccard(self, s1: list, s2: list) -> float:
        matches = sum(1 for x, y in zip(s1, s2) if x == y)
        return matches / len(s1)

    def is_duplicate(self, doc: str) -> bool:
        shingles = _shingles(doc, self.shingle_size)
        if not shingles:
            return False
        sig = self._minhash(shingles)
        # LSH: check candidates that share any band with the new doc
        candidates: Set[int] = set()
        for band in range(self._num_bands):
            key = tuple(sig[band * self._band_size:(band + 1) * self._band_size])
            for cid in self._bands[(band, key)]:
                candidates.add(cid)
            self._bands[(band, key)].append(self._next_id)
        for cid in candidates:
            if self._jaccard(sig, self._signatures[cid]) >= self.threshold:
                return True
        self._signatures[self._next_id] = sig
        self._next_id += 1
        return False

    def filter(self, docs: Iterable[str]) -> Iterator[str]:
        for doc in docs:
            if not self.is_duplicate(doc):
                yield doc
