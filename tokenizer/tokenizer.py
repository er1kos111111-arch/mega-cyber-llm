"""CyberTokenizer — the MEGA-CYBER LLM tokenizer.

A fully self-trained byte-level BPE tokenizer.  It loads its vocabulary
(``vocab.json``) and merges (``merges.json``), both produced by
:mod:`tokenizer.trainer`, and provides lossless encode/decode.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from .trainer import pre_tokenize
from .vocab import (
    NUM_BYTE_TOKENS,
    SPECIAL_TOKENS,
    build_special_token_map,
    initial_id_to_bytes,
    load_merges,
    load_vocab,
)


class CyberTokenizer:
    def __init__(self, config_path: str):
        """Load a tokenizer from ``tokenizer_config.json`` (dir may contain
        ``vocab.json`` and ``merges.json``).  ``config_path`` may be either
        the config file or the directory that contains it."""
        if os.path.isdir(config_path):
            config_path = os.path.join(config_path, "tokenizer_config.json")
        self.config_path = config_path
        self.dir = os.path.dirname(os.path.abspath(config_path))
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.vocab_path = self.config.get("vocab_file", "vocab.json")
        self.merges_path = self.config.get("merges_file", "merges.json")
        if not os.path.isabs(self.vocab_path):
            self.vocab_path = os.path.join(self.dir, self.vocab_path)
        if not os.path.isabs(self.merges_path):
            self.merges_path = os.path.join(self.dir, self.merges_path)

        self.vocab = load_vocab(self.vocab_path)
        self.special_map: Dict[str, int] = self.vocab["special_tokens"]
        self.merges: List[Tuple[int, int]] = load_merges(self.merges_path)

        self.vocab_size = self.vocab["vocab_size"]
        self.special_tokens = list(self.special_map.keys())

        # id -> bytes (base + special + merges)
        self.id_to_bytes: Dict[int, bytes] = initial_id_to_bytes(0)
        for pair in self.merges:
            a, b = pair
            self.id_to_bytes[len(self.id_to_bytes)] = self.id_to_bytes[a] + self.id_to_bytes[b]

        # (a, b) -> rank
        self.merge_rank: Dict[Tuple[int, int], int] = {pair: i for i, pair in enumerate(self.merges)}

        # convenience ids
        self.pad_token_id = self.special_map.get("<PAD>", 0)
        self.bos_token_id = self.special_map.get("<BOS>", 1)
        self.eos_token_id = self.special_map.get("<EOS>", 2)
        self.unk_token_id = self.special_map.get("<UNK>", 3)

        # word-level encode cache (big speedup on repeated vocabulary)
        self._encode_cache: Dict[bytes, List[int]] = {}
        self._cache_limit = 200_000

        # compiled matcher for special tokens (longest-first) so that strings
        # like "<USER>" are emitted as their special id, not byte-split
        self._special_sorted = sorted(self.special_tokens, key=len, reverse=True)
        self._special_re = re.compile("|".join(re.escape(t) for t in self._special_sorted))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.vocab_size

    def token_to_id(self, token: str) -> int:
        if token in self.special_map:
            return self.special_map[token]
        # single-byte token? (not applicable for text tokens, only specials here)
        raise KeyError(token)

    def id_to_token(self, token_id: int) -> str:
        if token_id < NUM_BYTE_TOKENS:
            return f"<0x{token_id:02X}>"
        if token_id in self.id_to_bytes:
            return self.id_to_bytes[token_id].decode("utf-8", errors="replace")
        return "<UNK>"

    # ------------------------------------------------------------------
    def _encode_word(self, word: bytes) -> List[int]:
        """Apply learned merges to a single word's byte sequence."""
        cached = self._encode_cache.get(word)
        if cached is not None:
            return cached
        ids = list(word)
        while len(ids) >= 2:
            best_rank = None
            best_pos = -1
            best_pair = None
            for i in range(len(ids) - 1):
                rank = self.merge_rank.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pos = i
                    best_pair = (ids[i], ids[i + 1])
            if best_pair is None:
                break
            a, b = best_pair
            merged = self.merge_rank[(a, b)] + NUM_BYTE_TOKENS + len(SPECIAL_TOKENS)
            ids = ids[:best_pos] + [merged] + ids[best_pos + 2:]
        if len(self._encode_cache) < self._cache_limit:
            self._encode_cache[word] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs (no BOS/EOS added).

        Strings containing special tokens (e.g. ``<USER>``) emit those
        special tokens as their dedicated ids.
        """
        if self._special_re.search(text):
            return self.encode_with_special(text)
        ids: List[int] = []
        for piece in pre_tokenize(text):
            if piece.isspace():
                ids.extend(list(piece.encode("utf-8")))
            else:
                ids.extend(self._encode_word(piece.encode("utf-8")))
        return ids

    def encode_with_special(self, text: str) -> List[int]:
        """Encode text, mapping special tokens to their dedicated ids."""
        ids: List[int] = []
        pos = 0
        for m in self._special_re.finditer(text):
            if m.start() > pos:
                ids.extend(self._encode_segment(text[pos:m.start()]))
            ids.append(self.special_map[m.group(0)])
            pos = m.end()
        if pos < len(text):
            ids.extend(self._encode_segment(text[pos:]))
        return ids

    def _encode_segment(self, text: str) -> List[int]:
        ids: List[int] = []
        for piece in pre_tokenize(text):
            if piece.isspace():
                ids.extend(list(piece.encode("utf-8")))
            else:
                ids.extend(self._encode_word(piece.encode("utf-8")))
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back into text."""
        special_ids = set(self.special_map.values())
        out = bytearray()
        for tid in ids:
            tid = int(tid)
            if skip_special_tokens and tid in special_ids:
                continue
            if tid in self.id_to_bytes:
                out.extend(self.id_to_bytes[tid])
            else:
                out.extend(f"<UNK>".encode("utf-8"))
        return out.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def tokenize(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode with optional ``<BOS>``/``<EOS>`` framing."""
        ids = self.encode(text)
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def pad(self, ids: List[int], max_len: int, pad_left: bool = False) -> List[int]:
        """Pad/truncate a token sequence to ``max_len``."""
        if len(ids) > max_len:
            return ids[:max_len]
        padding = [self.pad_token_id] * (max_len - len(ids))
        return padding + ids if pad_left else ids + padding

    def apply_chat_template(self, messages: List[dict], add_generation_prompt: bool = True) -> str:
        """Convert ``[{role, content}, ...]`` into the MC-LLM chat string."""
        role_map = {"system": "<SYSTEM>", "user": "<USER>",
                    "assistant": "<ASSISTANT>", "tool": "<TOOL>"}
        parts = []
        for m in messages:
            role = role_map.get(m.get("role", "user"), "<USER>")
            parts.append(f"{role}{m.get('content', '')}")
        if add_generation_prompt:
            parts.append("<ASSISTANT>")
        return "".join(parts)

    def tokenize_chat(self, messages: List[dict], add_bos: bool = True,
                      add_generation_prompt: bool = True) -> List[int]:
        """Tokenize a chat into ids using the special role tokens.

        Format: ``<BOS><USER>...<ASSISTANT>...<EOS>...<ASSISTANT>``
        """
        role_map = {"system": "<SYSTEM>", "user": "<USER>",
                    "assistant": "<ASSISTANT>", "tool": "<TOOL>"}
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        for m in messages:
            role = m.get("role", "user")
            ids.append(self.special_map[role_map.get(role, "<USER>")])
            ids.extend(self.encode(m.get("content", "")))
            if role == "assistant":
                ids.append(self.eos_token_id)
        if add_generation_prompt:
            ids.append(self.special_map["<ASSISTANT>"])
        return ids

    def tokenize_chat_with_labels(self, messages: List[dict]) -> Tuple[List[int], List[int]]:
        """Tokenize a chat and produce loss labels (supervise assistant only).

        Returns ``(input_ids, labels)`` where every non-assistant token is
        ``-100`` (ignored by cross-entropy).
        """
        role_map = {"system": "<SYSTEM>", "user": "<USER>",
                    "assistant": "<ASSISTANT>", "tool": "<TOOL>"}
        input_ids: List[int] = [self.bos_token_id]
        labels: List[int] = [-100]
        for m in messages:
            role = m.get("role", "user")
            content_ids = self.encode(m.get("content", ""))
            input_ids.append(self.special_map[role_map.get(role, "<USER>")])
            labels.append(-100)
            input_ids.extend(content_ids)
            if role == "assistant":
                labels.extend(content_ids)
                input_ids.append(self.eos_token_id)
                labels.append(self.eos_token_id)
            else:
                labels.extend([-100] * len(content_ids))
        return input_ids, labels

    def save_config(self, path: Optional[str] = None) -> None:
        """Persist the tokenizer config json (metadata only; vocab/merges are
        written by the trainer)."""
        cfg = {
            "name": "CyberTokenizer",
            "version": "1.0",
            "model_max_length": self.config.get("model_max_length", 8192),
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "vocab_file": self.config.get("vocab_file", "vocab.json"),
            "merges_file": self.config.get("merges_file", "merges.json"),
        }
        with open(path or self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
