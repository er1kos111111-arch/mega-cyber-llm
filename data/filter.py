"""Document filtering for the MC-LLM data pipeline.

Heuristic filters that run before/after cleaning:

* language detection (character-range + common-word scoring);
* length filters (too short / too long);
* quality filters (word/char ratio, mean word length, unique-word ratio);
* PII filtering (emails, phone numbers, addresses, credit cards);
* spam filtering (excessive punctuation, repeated phrases, ALL-CAPS ratio);
* repetition filtering (char/ngram repetition).
"""
from __future__ import annotations

import re
from typing import Iterable, Iterator, Optional

# ---------------------------------------------------------------------------
# language detection
# ---------------------------------------------------------------------------
_CYR = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
_LAT = set("abcdefghijklmnopqrstuvwxyz")

_RU_STOP = {"и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
            "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
            "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
            "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь",
            "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни",
            "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам"}

_EN_STOP = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there"}


def detect_language(text: str) -> str:
    """Return ``ru``, ``en`` or ``other`` based on script + stopwords."""
    words = re.findall(r"[а-яёa-z]+", text.lower())
    if not words:
        return "other"
    ru = sum(1 for w in words if w in _RU_STOP)
    en = sum(1 for w in words if w in _EN_STOP)
    cyr = sum(1 for ch in text.lower() if ch in _CYR)
    lat = sum(1 for ch in text.lower() if ch in _LAT)
    if cyr > lat * 2:
        return "ru"
    if ru > en and cyr > 0:
        return "ru"
    if en > 0 or lat > cyr * 2:
        return "en"
    return "other"


# ---------------------------------------------------------------------------
# length filters
# ---------------------------------------------------------------------------
def filter_length(text: str, min_chars: int = 50, max_chars: int = 100_000) -> bool:
    n = len(text)
    return min_chars <= n <= max_chars


def filter_word_count(text: str, min_words: int = 10, max_words: int = 50_000) -> bool:
    n = len(text.split())
    return min_words <= n <= max_words


# ---------------------------------------------------------------------------
# quality filters
# ---------------------------------------------------------------------------
def filter_word_char_ratio(text: str, min_ratio: float = 1.0) -> bool:
    words = text.split()
    if not words:
        return False
    mean_len = sum(len(w) for w in words) / len(words)
    return mean_len >= min_ratio


def filter_unique_words(text: str, min_ratio: float = 0.15) -> bool:
    words = text.split()
    if not words:
        return False
    return len(set(words)) / len(words) >= min_ratio


# ---------------------------------------------------------------------------
# PII / spam filters
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(\+?\d[\d\s()-]{7,}\d)")
_CREDIT_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def filter_pii(text: str) -> bool:
    """Return True if the text is *clean* (no PII)."""
    if _EMAIL_RE.search(text):
        return False
    if _PHONE_RE.search(text):
        return False
    if _CREDIT_RE.search(text):
        return False
    return True


def filter_spam(text: str) -> bool:
    """Return True if the text is *not* spam-like."""
    if text.count("!") > 0.1 * len(text):
        return False
    if text.count("?") > 0.1 * len(text):
        return False
    # ALL-CAPS ratio
    letters = [c for c in text if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
        return False
    # repeated punctuation runs
    if re.search(r"([!?.])\1{3,}", text):
        return False
    return True


def filter_repetition(text: str, max_char_ratio: float = 0.3) -> bool:
    """Reject documents dominated by a single repeated character."""
    if not text:
        return False
    from collections import Counter
    counts = Counter(text)
    top = counts.most_common(1)[0][1]
    return top / len(text) <= max_char_ratio


# ---------------------------------------------------------------------------
# combined pipeline
# ---------------------------------------------------------------------------
def filter_document(text: str, min_chars: int = 50, max_chars: int = 100_000,
                    detect_lang: bool = True, allowed_langs=None) -> bool:
    """Apply the full filter chain.  Returns True if document is kept."""
    if detect_lang:
        lang = detect_language(text)
        if allowed_langs is not None and lang not in allowed_langs:
            return False
    if not filter_length(text, min_chars, max_chars):
        return False
    if not filter_word_count(text):
        return False
    if not filter_word_char_ratio(text):
        return False
    if not filter_unique_words(text):
        return False
    if not filter_pii(text):
        return False
    if not filter_spam(text):
        return False
    if not filter_repetition(text):
        return False
    return True


def filter_documents(docs: Iterable[str], **kwargs) -> Iterator[str]:
    for doc in docs:
        if filter_document(doc, **kwargs):
            yield doc
