"""RL / post-training for MC-LLM.

Two pieces:

* ``RewardModel`` — a scalar head on top of the base model to score
  completions (trained on preference pairs);
* ``GRPO`` — Group Relative Policy Optimization: sample a group of
  completions per prompt, normalize rewards within the group, and update the
  policy with a clipped ratio objective (reference-model KL penalty).

This module is intentionally decoupled from the pretraining loop.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.architecture import MCLLM


class RewardModel(nn.Module):
    """Scalar reward head over the base model's final hidden state."""

    def __init__(self, base: MCLLM, hidden_size: int):
        super().__init__()
        self.base = base
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self._hidden(input_ids)
        return self.value_head(hidden)

    def _hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.base.embed_tokens(input_ids)
        for layer in self.base.layers:
            x = layer(x)
        x = self.base.norm(x)
        return x[:, -1, :]  # last token state


def grpo_step(policy: MCLLM, ref_policy: MCLLM, prompts: List[torch.Tensor],
              reward_fn, optimizer, epsilon: float = 0.2, beta: float = 0.04,
              num_samples: int = 4, device: str = "cpu", eos_token_id: int = 2,
              temperature: float = 1.0):
    """One GRPO update step."""
    from inference.generate import generate

    policy.train()
    advantages: List[float] = []
    sampled: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for prompt in prompts:
        prompt = prompt.to(device)
        group = []
        for _ in range(num_samples):
            out = generate(policy, prompt.unsqueeze(0), max_new_tokens=64,
                           eos_token_id=eos_token_id, temperature=temperature)
            completion = out[0]
            reward = reward_fn(completion)
            group.append((completion, reward))
        rewards = torch.tensor([g[1] for g in group], device=device)
        std = rewards.std() + 1e-8
        mean = rewards.mean()
        for completion, reward in group:
            sampled.append((completion, (reward - mean) / std))

    if not sampled:
        return 0.0

    total_loss = 0.0
    for completion, adv in sampled:
        input_ids = completion[:-1].unsqueeze(0)
        labels = completion[1:].unsqueeze(0)
        logits = policy(input_ids)
        logp = F.log_softmax(logits, dim=-1)
        per_token = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        log_prob = per_token.sum()
        with torch.no_grad():
            ref_logits = ref_policy(input_ids)
            ref_logp = F.log_softmax(ref_logits, dim=-1)
            ref_per = ref_logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            ref_log_prob = ref_per.sum()

        ratio = torch.exp(log_prob - ref_log_prob)
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        loss = -torch.min(ratio * adv, clipped * adv) + beta * (ratio - 1 - (log_prob - ref_log_prob))
        total_loss += loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item()
