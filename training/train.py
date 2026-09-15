"""Pretraining loop for MC-LLM.

Implements causal language modeling:

    input  = tokens[0 : n-1]
    target = tokens[1 : n]
    loss   = cross_entropy(logits, target)

with gradient accumulation, gradient clipping, mixed precision (FP16/BF16),
warmup+cosine LR schedule, weight decay, EMA (optional), logging, evaluation,
and checkpointing.
"""
from __future__ import annotations

import os
import math
import random
import time
from typing import Optional

import torch
import torch.nn as nn

from model.architecture import MCLLM
from model.config import RunConfig
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.scaler import build_scaler, autocast_context, nullcontext, resolve_dtype
from training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_optimizer_state,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: RunConfig) -> MCLLM:
    model = MCLLM(config.model)
    if config.model.tie_word_embeddings:
        model.embed_tokens.weight = model.lm_head.weight
    if config.distributed.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    dtype = _model_dtype(config)
    if dtype is not None and dtype != torch.float32 and torch.cuda.is_available():
        model = model.to(dtype)
    return model


def _model_dtype(config: RunConfig) -> Optional[torch.dtype]:
    from training.scaler import resolve_dtype
    dt = resolve_dtype(config.training.dtype or config.model.dtype or "float32")
    if dt == "bfloat16":
        return torch.bfloat16
    if dt == "float16":
        return torch.float16
    return torch.float32


def _to_device(model, device):
    if device == "cuda" and torch.cuda.is_available():
        return model.cuda()
    return model


def train(
    config: RunConfig,
    train_loader=None,
    eval_loader=None,
    resume: bool = True,
    log_fn=print,
):
    """Run pretraining.  Returns a summary dict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config.training.seed)

    model = build_model(config)
    model = _to_device(model, device)

    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log_fn(f"[MC-LLM] device: {name} ({vram:.1f} GB), "
               f"dtype={resolve_dtype(config.training.dtype)}")
    else:
        log_fn("[MC-LLM] device: cpu")

    n_params = model.num_parameters()
    log_fn(f"[MC-LLM] parameters: {n_params:,}")

    optimizer = build_optimizer(
        model.parameters(),
        name=getattr(config.training, "optimizer", "adamw"),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=tuple(config.training.betas),
        eps=config.training.eps,
    )
    scheduler = build_scheduler(
        "warmup_cosine", optimizer,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        min_lr=config.training.min_learning_rate,
    )
    scaler = build_scaler(config.training.dtype)

    # optional EMA
    ema_model = None
    if config.training.use_ema:
        ema_model = _EMA(model, decay=config.training.ema_decay)

    start_step = 0
    tokens_seen = 0
    if resume:
        latest = os.path.join(config.out_dir, "latest.json")
        if os.path.exists(latest):
            start_step, meta = load_checkpoint(model, config.out_dir)
            load_optimizer_state(optimizer, scheduler, config.out_dir, start_step)
            tokens_seen = meta.get("tokens_seen", 0)
            log_fn(f"[MC-LLM] resumed from step {start_step}, tokens {tokens_seen:,}")

    # gradient accumulation: batch size in *sequences*
    micro_batch = config.training.micro_batch_size
    grad_accum = config.training.gradient_accumulation_steps

    autocast = autocast_context(config.training.dtype, device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = start_step
    total_loss = 0.0
    running = 0
    t0 = time.time()
    last_log = time.time()
    last_reported_loss = float("nan")

    # DataLoader must be provided (built by scripts/train.py), else create default
    if train_loader is None:
        from data.dataset import build_dataloader
        train_loader = build_dataloader(
            config.data_dir, config.training.sequence_length, micro_batch,
            seed=config.training.seed)

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast:
            logits = model(inputs)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, : shift_logits.size(1)].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )
            loss = loss / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        tokens_seen += inputs.numel()
        total_loss += loss.item() * grad_accum
        running += 1

        if (running % grad_accum) == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema_model is not None:
                ema_model.update(model)
            step += 1
            scheduler.step(step)

            if step % config.training.log_interval == 0:
                now = time.time()
                dt = now - last_log
                last_log = now
                avg_loss = total_loss / max(1, config.training.log_interval)
                last_reported_loss = avg_loss
                tok_per_sec = (config.training.global_batch_size) / dt if dt > 0 else 0
                log_fn(
                    f"step {step}/{config.training.max_steps} "
                    f"loss {avg_loss:.4f} lr {scheduler.get_lr(step):.2e} "
                    f"tokens {tokens_seen:,} ({tok_per_sec:.0f} tok/s)"
                )
                total_loss = 0.0

            if step % config.training.save_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, step, config.out_dir, config,
                    tokens_seen=tokens_seen, loss=avg_loss)
                log_fn(f"[MC-LLM] saved checkpoint at step {step}")

            if step % config.training.eval_interval == 0 and eval_loader is not None:
                val_loss = evaluate(model, eval_loader, device)
                log_fn(f"[MC-LLM] validation loss {val_loss:.4f} ppl {_ppl(val_loss):.2f}")

            if step >= config.training.max_steps:
                break

    # final save
    if math.isnan(last_reported_loss):
        last_reported_loss = total_loss / max(1, running)
    meta = save_checkpoint(model, optimizer, scheduler, step, config.out_dir, config,
                           tokens_seen=tokens_seen, loss=last_reported_loss)
    return {
        "parameters": n_params,
        "final_step": step,
        "tokens_seen": tokens_seen,
        "loss": last_reported_loss,
        "checkpoint": meta.get("checkpoint"),
    }


@torch.no_grad()
def evaluate(model, eval_loader, device="cpu"):
    model.eval()
    total = 0.0
    n = 0
    for inputs, targets in eval_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, : shift_logits.size(1)].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1), ignore_index=-100)
        total += loss.item()
        n += 1
    model.train()
    return total / max(1, n)


def _ppl(loss: float) -> float:
    import math
    return math.exp(min(loss, 100))


class _EMA:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.named_parameters()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.named_parameters():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)
