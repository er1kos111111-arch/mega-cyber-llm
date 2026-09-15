"""From-scratch Byte-Pair-Encoding trainer for CyberTokenizer.

This is a self-contained BPE implementation (no SentencePiece, no HF
``tokenizers``, no third-party vocabulary).  It learns merges directly
from the project's own corpus.

Design
------
* Pre-tokenization splits text into *word* and *whitespace* runs
  (non-space vs space).  Whitespace runs stay as raw byte tokens; merges
  are only learned/applied *inside* words.  This keeps the algorithm
  deterministic and fast while still handling arbitrary Unicode.
* Base symbols are the 256 byte values (so every byte is representable,
  making the tokenizer lossless on any UTF-8 text).
* Training counts adjacent-pair frequencies across the corpus and greedily
  merges the most frequent pair until ``vocab_size`` is reached.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple

from .vocab import NUM_BYTE_TOKENS, SPECIAL_TOKENS

_TOKEN_SPLIT = re.compile(r"(\s+|\S+)")


def pre_tokenize(text: str) -> Iterator[str]:
    """Yield alternating whitespace / word pieces."""
    for piece in _TOKEN_SPLIT.findall(text):
        if piece:
            yield piece


def _bytes_to_ids(b: bytes) -> List[int]:
    return list(b)


class CyberTokenizerTrainer:
    def __init__(self, vocab_size: int = 32768, min_pair_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_pair_frequency = min_pair_frequency
        self.merges: List[Tuple[int, int]] = []
        # word (as bytes) -> current symbol (token-id) sequence
        self.word_symbols: Dict[bytes, List[int]] = {}
        # pair -> total frequency
        self.pair_counts: Counter = Counter()
        self.next_id = NUM_BYTE_TOKENS + len(SPECIAL_TOKENS)

    # ------------------------------------------------------------------
    def _count_word(self, word: bytes, freq: int) -> None:
        if word in self.word_symbols:
            return
        symbols = _bytes_to_ids(word)
        self.word_symbols[word] = symbols
        self._increment_pairs(symbols, freq)

    def _increment_pairs(self, symbols: List[int], delta: int) -> None:
        for i in range(len(symbols) - 1):
            self.pair_counts[(symbols[i], symbols[i + 1])] += delta

    # ------------------------------------------------------------------
    def _add_to_corpus(self, texts: Iterable[str]) -> None:
        word_freq: Counter = Counter()
        for text in texts:
            for piece in pre_tokenize(text):
                if piece.isspace():
                    continue
                word_freq[piece.encode("utf-8")] += 1
        for word, freq in word_freq.items():
            self._count_word(word, freq)

    # ------------------------------------------------------------------
    def _merge_pair(self, a: int, b: int, new_id: int) -> None:
        """Replace every adjacent ``(a, b)`` occurrence with ``new_id``."""
        for word, symbols in list(self.word_symbols.items()):
            if len(symbols) < 2:
                continue
            new_symbols: List[int] = []
            i = 0
            n = len(symbols)
            changed = False
            while i < n:
                if i + 1 < n and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(new_id)
                    i += 2
                    changed = True
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            if changed:
                # recompute pair counts for this word
                old = symbols
                self.word_symbols[word] = new_symbols
                self._increment_pairs(old, -1)
                self._increment_pairs(new_symbols, +1)

    # ------------------------------------------------------------------
    def train(self, texts: Iterable[str], min_frequency: int = 2) -> List[Tuple[int, int]]:
        """Learn merges.  Returns the ordered list of ``(a, b)`` pairs."""
        self._add_to_corpus(texts)

        target_merges = self.vocab_size - (NUM_BYTE_TOKENS + len(SPECIAL_TOKENS))

        with_progress = True
        pbar = None
        if with_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=target_merges, desc="BPE merges")
            except Exception:
                pbar = None

        while len(self.merges) < target_merges:
            if not self.pair_counts:
                break
            (a, b), freq = self.pair_counts.most_common(1)[0]
            if freq < self.min_pair_frequency:
                break
            new_id = self.next_id
            self.merges.append((a, b))
            self.next_id += 1
            # remove the merged pair from counts before updating
            del self.pair_counts[(a, b)]
            self._merge_pair(a, b, new_id)
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()
        return self.merges

    def build_merges_rank(self) -> Dict[Tuple[int, int], int]:
        """Map ``(a, b) -> rank`` (lower rank = merged earlier = higher priority)."""
        return {pair: rank for rank, pair in enumerate(self.merges)}
