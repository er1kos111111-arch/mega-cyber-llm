"""Vocabulary construction for CyberTokenizer.

The vocabulary is built from scratch:

* IDs ``0..255`` are single *bytes* (byte-level BPE foundation);
* IDs ``256..256+len(SPECIAL_TOKENS)-1`` are the special control tokens;
* IDs after that are learned BPE *merge* tokens.

Each token maps to a ``bytes`` object.  Special tokens map to their
literal string encoded as UTF-8 so that decode round-trips them cleanly.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

# Order matters: this defines the fixed special-token IDs.
SPECIAL_TOKENS: List[str] = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
    "<USER>",
    "<ASSISTANT>",
    "<SYSTEM>",
    "<TOOL>",
    "<THINK>",
    "<END_THINK>",
    "<CODE>",
    "<END_CODE>",
]

NUM_BYTE_TOKENS = 256


def build_special_token_map() -> Dict[str, int]:
    return {tok: NUM_BYTE_TOKENS + i for i, tok in enumerate(SPECIAL_TOKENS)}


def special_token_bytes(token: str) -> bytes:
    """Bytes that a special token decodes to (its own literal string)."""
    return token.encode("utf-8")


def initial_id_to_bytes(num_merges: int) -> Dict[int, bytes]:
    """Rebuild ``id -> bytes`` from the base byte tokens + special tokens."""
    table: Dict[int, bytes] = {}
    for b in range(NUM_BYTE_TOKENS):
        table[b] = bytes([b])
    for i, tok in enumerate(SPECIAL_TOKENS):
        table[NUM_BYTE_TOKENS + i] = special_token_bytes(tok)
    return table


def save_vocab(path: str, special_map: Dict[str, int], merges: List[Tuple[int, int]],
               total_vocab_size: int) -> None:
    payload = {
        "version": "1.0",
        "num_byte_tokens": NUM_BYTE_TOKENS,
        "special_tokens": special_map,
        "vocab_size": total_vocab_size,
        "num_merges": len(merges),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_merges(path: str, merges: List[Tuple[int, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merges, f, ensure_ascii=False)


def load_vocab(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_merges(path: str) -> List[Tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [tuple(pair) for pair in raw]
