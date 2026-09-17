"""Supervised Fine-Tuning (SFT) for MC-LLM.

Turns a pretrained model into a conversational assistant by training on chat
data (``<USER>`` / ``<ASSISTANT>`` / ``<SYSTEM>`` tokens) with loss computed
**only** over assistant tokens.

This is a full, production-quality loop: mixed precision, warmup+cosine LR,
gradient clipping, checkpointing, and a validation loss.
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn

from data.sft_data import SFTDataset, collate_sft
from model.architecture import MCLLM
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.scaler import build_scaler, autocast_context, nullcontext, resolve_dtype
from training.checkpoint import save_checkpoint


def sft_train(
    model: MCLLM,
    tokenizer,
    messages_list: List[List[dict]],
    out_dir: str = "checkpoints_sft",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    dtype: str = "bfloat16",
    max_length: int = 1024,
    seed: int = 42,
    eval_frac: float = 0.05,
    log_every: int = 10,
    save_every: int = 500,
    resume: bool = True,
    device: str = "cuda",
    log_fn=print,
) -> dict:
    """Run SFT.  Returns a summary dict.  Saves checkpoints periodically and
    resumes from the last one if ``resume`` is set."""
    torch.manual_seed(seed)
    device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    resolved = resolve_dtype(dtype)
    if resolved != "float32" and device == "cuda":
        model = model.to(torch.bfloat16 if resolved == "bfloat16" else torch.float16)
    model = model.to(device)
    if model.config.num_layers >= 16:
        model.enable_gradient_checkpointing()

    # never exceed the model's positional-encoding context
    max_length = min(max_length, model.config.max_position_embeddings)

    n = len(messages_list)
    split = max(1, int(n * eval_frac))
    train_msgs = messages_list[split:]
    val_msgs = messages_list[:split]
    train_ds = SFTDataset(train_msgs, tokenizer, max_length=max_length)
    val_ds = SFTDataset(val_msgs, tokenizer, max_length=max_length)
    log_fn(f"[SFT] {len(train_ds):,} train examples, {len(val_ds):,} val examples")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_sft)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, collate_fn=collate_sft)

    optimizer = build_optimizer(model.parameters(), name="adamw", lr=lr,
                                weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    scheduler = build_scheduler("warmup_cosine", optimizer,
                                warmup_steps=max(1, total_steps // 20),
                                max_steps=total_steps, min_lr=lr * 0.1)
    scaler = build_scaler(dtype)
    autocast = autocast_context(dtype, device)

    os.makedirs(out_dir, exist_ok=True)

    # resume from a previous SFT checkpoint
    start_step = 0
    if resume:
        latest = os.path.join(out_dir, "latest.json")
        if os.path.exists(latest):
            from training.checkpoint import load_checkpoint
            start_step, _ = load_checkpoint(model, out_dir)
            log_fn(f"[SFT] resumed from step {start_step}")

    from model.config import RunConfig
    run = RunConfig(model=model.config, out_dir=out_dir)

    model.train()
    step = start_step
    running_loss = 0.0
    loader_iter = iter(train_loader)

    while step < total_steps:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)
        x, y = x.to(device), y.to(device)
        with autocast:
            logits = model(x)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if scaler is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step(step)
        step += 1
        running_loss += loss.item()

        if step % log_every == 0:
            avg = running_loss / log_every
            log_fn(f"[SFT] step {step}/{total_steps} "
                   f"loss {avg:.4f} lr {scheduler.get_lr(step):.2e}")
            running_loss = 0.0

        if save_every and step % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step, out_dir, run,
                            loss=loss.item())
            log_fn(f"[SFT] saved checkpoint at step {step}")

    # validation
    val_loss = _eval_sft(model, val_loader, device)

    import math
    summary = {
        "sft_loss": val_loss,
        "sft_perplexity": round(math.exp(min(val_loss, 100)), 2),
        "epochs": epochs,
        "steps": step,
    }
    log_fn(f"[SFT] validation loss {val_loss:.4f}, ppl {summary['sft_perplexity']}")

    # save final checkpoint (meta carries the model config)
    meta = save_checkpoint(model, optimizer, scheduler, step, out_dir, run,
                           loss=val_loss)
    summary["checkpoint"] = meta.get("checkpoint")
    return summary


@torch.no_grad()
def _eval_sft(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = y[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100)
        total += loss.item()
        n += 1
    model.train()
    return total / max(1, n)
