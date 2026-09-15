"""Automatic quality control for generated conversations.

Each generated dialogue is checked for: format validity, role alternation,
empty messages, excessive repetition, degenerate loops, and broken Unicode.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

_ROLE_RE = re.compile(r"^(USER|ASSISTANT):\s*(.*)$", re.MULTILINE)


def split_turns(text: str) -> List[str]:
    """Return the list of message bodies (without the USER:/ASSISTANT: prefix)."""
    return [m.group(2).strip() for m in _ROLE_RE.finditer(text)]


def check_format(text: str) -> bool:
    """Every line must alternate USER:/ASSISTANT: and start with USER:."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    if not lines[0].startswith("USER:"):
        return False
    prev = None
    for ln in lines:
        if ln.startswith("USER:"):
            cur = "USER"
        elif ln.startswith("ASSISTANT:"):
            cur = "ASSISTANT"
        else:
            return False
        if prev == cur:
            return False
        prev = cur
    return True


def check_min_turns(text: str, min_turns: int = 2) -> bool:
    return len(split_turns(text)) >= min_turns


def check_no_empty(text: str) -> bool:
    return all(t for t in split_turns(text))


def check_no_broken_roles(text: str) -> bool:
    return text.count("USER:") >= 1 and "ASSISTANT:" in text


def check_repetition(text: str, max_dup_ratio: float = 0.5,
                     max_ngram: int = 4) -> bool:
    """Reject dialogues where a single message or n-gram dominates."""
    turns = split_turns(text)
    if not turns:
        return False
    # identical message bodies within the dialogue
    counts = Counter(turns)
    if max(counts.values()) > max(2, len(turns) * max_dup_ratio):
        return False
    # excessive repetition of a short n-gram across the whole text
    tokens = text.lower().split()
    if len(tokens) > 20:
        grams = Counter(tuple(tokens[i:i + max_ngram])
                        for i in range(len(tokens) - max_ngram + 1))
        top = grams.most_common(1)[0][1]
        if top / max(1, len(tokens)) > 0.25:
            return False
    return True


def check_unicode(text: str) -> bool:
    return "\ufffd" not in text and all(ord(c) < 0xD800 or ord(c) > 0xDFFF for c in text)


def check_message_len(text: str, max_chars: int = 300) -> bool:
    return all(len(t) <= max_chars for t in split_turns(text))


def check_dialogue(text: str, min_turns: int = 2, max_chars: int = 300) -> bool:
    """Full quality gate.  Returns True if the dialogue passes."""
    return (check_format(text)
            and check_min_turns(text, min_turns)
            and check_no_empty(text)
            and check_no_broken_roles(text)
            and check_repetition(text)
            and check_unicode(text)
            and check_message_len(text, max_chars))


def quality_report(text: str) -> Dict[str, bool]:
    """Diagnostic breakdown (used by tests / debugging)."""
    return {
        "format": check_format(text),
        "min_turns": check_min_turns(text),
        "no_empty": check_no_empty(text),
        "roles": check_no_broken_roles(text),
        "repetition": check_repetition(text),
        "unicode": check_unicode(text),
    }
