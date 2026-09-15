"""SFT data preparation for MC-LLM.

Turns conversational data (real PersonaChat dialogues + synthetic Russian
conversations) into chat ``messages`` and tokenizes them into
``(input_ids, labels)`` pairs with assistant-only loss masking.
"""
from __future__ import annotations

import json
import os
from typing import Iterator, List, Tuple

import torch
from torch.utils.data import Dataset


def text_to_messages(text: str) -> List[dict]:
    """Convert ``USER: ... / ASSISTANT: ...`` text into chat messages."""
    from data.conversation.quality import split_turns
    turns = split_turns(text)
    messages = []
    for i, t in enumerate(turns):
        if not t:
            continue
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": t})
    return messages


def messages_from_texts(texts: Iterator[str]) -> Iterator[List[dict]]:
    for text in texts:
        msgs = text_to_messages(text)
        if len(msgs) >= 2 and any(m["role"] == "assistant" for m in msgs):
            yield msgs


def build_sft_jsonl(texts: Iterator[str], out_path: str) -> int:
    """Write ``{"messages": [...]}`` lines to a JSONL.  Returns count."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for msgs in messages_from_texts(texts):
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n += 1
    return n


class SFTDataset(Dataset):
    """Map-style dataset of tokenized chat examples.

    Each example is ``(input_ids, labels)`` where labels are ``-100`` for
    every non-assistant token (so cross-entropy only supervises the
    assistant's replies).
    """

    def __init__(self, messages_list: List[List[dict]], tokenizer,
                 max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Tuple[List[int], List[int]]] = []
        for msgs in messages_list:
            ids, labels = tokenizer.tokenize_chat_with_labels(msgs)
            if len(ids) > max_length:
                ids = ids[:max_length]
                labels = labels[:max_length]
            self.examples.append((ids, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ids, labels = self.examples[idx]
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))


def collate_sft(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Pad a batch of (input_ids, labels) to equal length."""
    pad = batch[0][0].device
    max_len = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), max_len), 0, dtype=torch.long)  # 0 = <PAD>
    y = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        x[i, :len(ids)] = ids
        y[i, :len(labels)] = labels
    return x, y


def load_messages_from_jsonl(path: str) -> List[List[dict]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            if msgs:
                out.append(msgs)
    return out


def build_sft_messages(synthetic_n: int = 20000, persona_max_rows: int = 0,
                       seed: int = 0, raw_dir: str = "data/raw") -> List[List[dict]]:
    """Assemble SFT messages from synthetic Russian conversations and
    (optionally) the real PersonaChat dialogue dataset."""
    messages: List[List[dict]] = []

    # 1. synthetic Russian conversations (fast, large, no network)
    from data.conversation.generator import DialogueGenerator
    gen = DialogueGenerator(seed=seed)
    for _ in range(synthetic_n):
        d = gen.generate()
        msgs = text_to_messages(d["text"])
        if len(msgs) >= 2:
            messages.append(msgs)
    print(f"[sft-data] synthetic: {len(messages):,} examples")

    # 2. real PersonaChat dialogues (optional)
    if persona_max_rows > 0:
        try:
            from data.download import download_preset
            download_preset("persona_chat_ru", out_dir=raw_dir, max_rows=persona_max_rows)
            path = os.path.join(raw_dir, "persona_chat_ru.jsonl")
            before = len(messages)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    msgs = text_to_messages(obj.get("text", ""))
                    if len(msgs) >= 2:
                        messages.append(msgs)
            print(f"[sft-data] persona-chat: +{len(messages) - before:,} examples")
        except Exception as e:
            print(f"[sft-data] persona-chat skipped: {e}")

    return messages
