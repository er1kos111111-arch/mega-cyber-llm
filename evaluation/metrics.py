"""Evaluation metrics for MC-LLM."""
from __future__ import annotations

import math
from collections import Counter
from typing import List


def perplexity(loss: float) -> float:
    return math.exp(min(float(loss), 700))


def repetition_ratio(tokens: List[int], n: int = 4) -> float:
    """Fraction of n-grams that repeat (a high value indicates looping)."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def distinct_ratio(tokens: List[int]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if prediction.strip() == reference.strip() else 0.0


def contains_any(prediction: str, answers: List[str]) -> float:
    p = prediction.lower()
    return 1.0 if any(a.lower() in p for a in answers) else 0.0


def language_guess(text: str) -> str:
    """Quick script-based language guess for multilingual reporting."""
    cyr = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    return "ru" if cyr > lat else ("en" if lat > cyr else "other")
