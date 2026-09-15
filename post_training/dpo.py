"""Direct Preference Optimization (DPO) for MC-LLM.

DPO trains the policy directly on (prompt, chosen, rejected) triples using
the closed-form objective, without a separate reward model.

    loss = -log σ(β·[logπ(y_w|x)/π_ref(y_w|x) − logπ(y_l|x)/π_ref(y_l|x)])

The reference model is a frozen copy of the SFT model.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sequence log-probability (sum over tokens, ignoring -100)."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    logp = F.log_softmax(shift_logits, dim=-1)
    per_token = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    return (per_token * mask).sum(dim=-1)


def dpo_loss(policy, ref_policy, x_w, y_w, x_l, y_l, beta: float = 0.1) -> torch.Tensor:
    with torch.no_grad():
        ref_w = log_probs_from_logits(ref_policy(x_w), y_w)
        ref_l = log_probs_from_logits(ref_policy(x_l), y_l)
    pi_w = log_probs_from_logits(policy(x_w), y_w)
    pi_l = log_probs_from_logits(policy(x_l), y_l)

    log_ratio_w = pi_w - ref_w
    log_ratio_l = pi_l - ref_l
    return -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()


def dpo_train(policy, ref_policy, pairs, optimizer, steps: int,
              beta: float = 0.1, device: str = "cpu"):
    """Minimal DPO loop.  ``pairs`` = list of (x_w, y_w, x_l, y_l) tensors."""
    policy.train()
    ref_policy.eval()
    for step in range(steps):
        x_w, y_w, x_l, y_l = pairs[step % len(pairs)]
        x_w, y_w = x_w.to(device), y_w.to(device)
        x_l, y_l = x_l.to(device), y_l.to(device)
        optimizer.zero_grad()
        loss = dpo_loss(policy, ref_policy, x_w, y_w, x_l, y_l, beta)
        loss.backward()
        optimizer.step()
    return policy
