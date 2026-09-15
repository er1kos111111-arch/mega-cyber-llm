"""Supervised Fine-Tuning (SFT) for MC-LLM.

Turns the pretrained model into a chat model by training on the
``SYSTEM/USER/ASSISTANT`` dialogue format.  Loss is computed only over
assistant tokens (prompt tokens are masked out).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def format_chat(turn: dict, tokenizer) -> str:
    role = turn.get("role", "user")
    token = {"system": "<SYSTEM>", "user": "<USER>",
             "assistant": "<ASSISTANT>", "tool": "<TOOL>"}.get(role, "<USER>")
    return f"{token}\n{turn.get('content', '')}\n"


def build_sft_example(messages: List[dict], tokenizer) -> dict:
    """Encode a conversation, masking every token that is not assistant output."""
    full = "".join(format_chat(m, tokenizer) for m in messages)
    full += "<ASSISTANT>\n"

    input_ids: List[int] = []
    labels: List[int] = []
    for m in messages:
        prefix = format_chat(m, tokenizer)
        ids = tokenizer.encode(prefix)
        input_ids.extend(ids)
        if m.get("role") == "assistant":
            labels.extend(ids)
        else:
            labels.extend([-100] * len(ids))
    # trailing assistant marker
    tail = tokenizer.encode("<ASSISTANT>\n")
    input_ids.extend(tail)
    labels.extend([-100] * len(tail))
    return {"input_ids": input_ids, "labels": labels}


def sft_loss(model, batch_input_ids: torch.Tensor, batch_labels: torch.Tensor,
             pad_token_id: int) -> torch.Tensor:
    logits = model(batch_input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch_labels[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1), ignore_index=-100)


def sft_train(model, examples, tokenizer, optimizer, steps: int,
              batch_size: int = 4, device: str = "cpu"):
    """Minimal SFT loop over in-memory examples (see scripts for full version)."""
    model.train()
    dataset = [build_sft_example(msgs, tokenizer) for msgs in examples]
    import random
    random.shuffle(dataset)

    for step in range(steps):
        batch = dataset[step % len(dataset):][:batch_size]
        max_len = max(len(e["input_ids"]) for e in batch)
        pad = tokenizer.pad_token_id
        x = torch.full((len(batch), max_len), pad, dtype=torch.long, device=device)
        y = torch.full((len(batch), max_len), -100, dtype=torch.long, device=device)
        for i, e in enumerate(batch):
            x[i, :len(e["input_ids"])] = torch.tensor(e["input_ids"])
            y[i, :len(e["labels"])] = torch.tensor(e["labels"])
        optimizer.zero_grad()
        loss = sft_loss(model, x, y, pad)
        loss.backward()
        optimizer.step()
    return model
