"""Autoregressive generation engine for MC-LLM.

Implements a real, token-by-token decoder with:

* greedy decoding;
* temperature sampling;
* top-k;
* top-p (nucleus);
* repetition penalty;
* min-p;
* stop tokens;
* max tokens;
* streaming (generator interface);
* optional KV cache.

No external inference framework is used — sampling is implemented here.
"""
from __future__ import annotations

import math
from typing import Generator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from model.architecture import MCLLM
from inference.kv_cache import make_cache


@torch.no_grad()
def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    previous_ids: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a single token from the final-row logits."""
    logits = logits[..., -1, :]  # (batch, vocab)

    if repetition_penalty != 1.0 and previous_ids is not None:
        # penalize tokens already generated in this sequence
        for i in range(logits.size(0)):
            seen = set(previous_ids[i].tolist())
            for tok in seen:
                if logits[i, tok] > 0:
                    logits[i, tok] /= repetition_penalty
                else:
                    logits[i, tok] *= repetition_penalty

    if temperature <= 0.0 or temperature is None:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        top_prob = probs.max(dim=-1, keepdim=True).values
        threshold = min_p * top_prob
        logits = logits.masked_fill(probs < threshold, float("-inf"))

    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        kth = v[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        keep = torch.zeros_like(logits, dtype=torch.bool)
        keep.scatter_(1, sorted_idx, ~remove)
        logits = logits.masked_fill(~keep, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def generate(
    model: MCLLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    eos_token_id: int = 2,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    min_p: float = 0.0,
    stop_token_ids: Optional[List[int]] = None,
    seed: Optional[int] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Generate up to ``max_new_tokens`` tokens.  Returns full token sequence."""
    return list(generate_stream(
        model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty, min_p=min_p,
        stop_token_ids=stop_token_ids, seed=seed, use_cache=use_cache,
    ))[-1]


@torch.no_grad()
def generate_stream(
    model: MCLLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    eos_token_id: int = 2,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    min_p: float = 0.0,
    stop_token_ids: Optional[List[int]] = None,
    seed: Optional[int] = None,
    use_cache: bool = True,
) -> Generator[torch.Tensor, None, None]:
    """Generator that yields the growing token sequence after each new token."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    batch = input_ids.size(0)
    generated = input_ids.clone()

    generator = None
    if seed is not None:
        if device == "cuda":
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

    stop_ids = set(stop_token_ids or [])
    if eos_token_id is not None:
        stop_ids.add(eos_token_id)

    cache = make_cache(model, batch_size=batch) if use_cache else None

    # prefill: logits for the last prompt token
    if cache is not None:
        pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
        logits = model(input_ids, position_ids=pos, kv_cache=cache)
        next_pos = input_ids.size(1)
    else:
        pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
        logits = model(input_ids, position_ids=pos)
        next_pos = input_ids.size(1)

    for _ in range(max_new_tokens):
        prev_ids = generated if repetition_penalty != 1.0 else None
        next_tok = _sample_token(logits, temperature, top_k, top_p, min_p,
                                 repetition_penalty, prev_ids, generator)
        generated = torch.cat([generated, next_tok.unsqueeze(-1)], dim=-1)
        yield generated

        if stop_ids and all(int(t) in stop_ids for t in next_tok.tolist()):
            break

        # compute logits for the following token
        if cache is not None:
            current = generated[:, -1:]
            pos = torch.full((batch, 1), next_pos, device=device)
            logits = model(current, position_ids=pos, kv_cache=cache)
            next_pos += 1
        else:
            pos = torch.arange(generated.size(1), device=device).unsqueeze(0)
            logits = model(generated, position_ids=pos)


def greedy_decode(model, input_ids, max_new_tokens=64, eos_token_id=2):
    return generate(model, input_ids, max_new_tokens=max_new_tokens,
                    temperature=0.0, eos_token_id=eos_token_id)
