"""Generation quality metrics for MC-LLM.

Detect the failure modes that make output look "broken": repetition loops,
broken Unicode, fragmented words, invalid/control characters, and absurd
symbol density.  These are heuristic — they flag obviously-bad text, not
subtle grammar errors.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import List


def repetition_ratio(tokens: List[int], n: int = 4) -> float:
    """Fraction of repeated n-grams (high = looping)."""
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(grams)


def max_run_ratio(tokens: List[int]) -> float:
    """Longest run of a single token, as a fraction of the sequence."""
    if not tokens:
        return 0.0
    best = cur = 1
    for i in range(1, len(tokens)):
        cur = cur + 1 if tokens[i] == tokens[i - 1] else 1
        best = max(best, cur)
    return best / len(tokens)


def invalid_char_ratio(text: str) -> float:
    """Fraction of characters that are replacement chars / control chars."""
    if not text:
        return 0.0
    bad = sum(1 for c in text if c == "\ufffd" or (ord(c) < 32 and c not in "\n\t"))
    return bad / len(text)


def non_letter_symbol_ratio(text: str) -> float:
    """Fraction of characters that are neither letters, digits, whitespace,
    nor common punctuation — flags symbol soup like ``#@$^&``."""
    if not text:
        return 0.0
    allowed = set(".,!?;:()«»„“\"'—-… \n\t0123456789")
    weird = sum(1 for c in text
                if not (c.isalnum() or c.isspace() or c in allowed))
    return weird / len(text)


def word_fragment_ratio(text: str) -> float:
    """Fraction of 1-char words — high values suggest fragmented output."""
    words = [w for w in re.findall(r"[^\s]+", text) if any(ch.isalpha() for ch in w)]
    if not words:
        return 0.0
    return sum(1 for w in words if len(w) == 1) / len(words)


def assess_generation(text: str, tokens: List[int]) -> dict:
    """Return a dict of quality metrics for a generated reply."""
    return {
        "length": len(text),
        "tokens": len(tokens),
        "repetition_ratio": round(repetition_ratio(tokens), 3),
        "max_run_ratio": round(max_run_ratio(tokens), 3),
        "invalid_char_ratio": round(invalid_char_ratio(text), 4),
        "symbol_ratio": round(non_letter_symbol_ratio(text), 4),
        "word_fragment_ratio": round(word_fragment_ratio(text), 3),
    }


def is_broken(metrics: dict) -> bool:
    """Heuristic flag: is this generation obviously broken/garbage?"""
    return (metrics["invalid_char_ratio"] > 0.02
            or metrics["symbol_ratio"] > 0.05
            or metrics["repetition_ratio"] > 0.4
            or metrics["max_run_ratio"] > 0.5
            or metrics["word_fragment_ratio"] > 0.3)
