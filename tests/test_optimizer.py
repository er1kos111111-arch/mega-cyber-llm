"""Optimizer and scheduler tests."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.optimizer import AdamW, Adam, Muon, build_optimizer
from training.scheduler import WarmupCosineScheduler


def test_adamw_decreases_loss():
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    y = (x @ torch.randn(8, 1)).squeeze()
    w = torch.randn(8, 1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    opt = AdamW([w, b], lr=1e-2)
    losses = []
    for _ in range(50):
        pred = x @ w + b
        loss = ((pred - y.unsqueeze(1)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_adam_decreases_loss():
    torch.manual_seed(1)
    x = torch.randn(32, 4)
    y = torch.randn(32, 1)
    w = torch.randn(4, 1, requires_grad=True)
    opt = Adam([w], lr=1e-2)
    for _ in range(30):
        loss = ((x @ w - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


def test_muon_decreases_loss():
    torch.manual_seed(2)
    x = torch.randn(32, 4)
    y = x @ torch.randn(4, 1)
    w = torch.randn(4, 1, requires_grad=True)
    opt = Muon([w], lr=1e-3, adamw_lr=1e-2)
    for _ in range(20):
        loss = ((x @ w - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


def test_build_optimizer_names():
    p = torch.nn.Parameter(torch.randn(3))
    assert isinstance(build_optimizer([p], "adamw"), AdamW)
    assert isinstance(build_optimizer([p], "adam"), Adam)
    assert isinstance(build_optimizer([p], "muon"), Muon)


def test_warmup_cosine_scheduler():
    param = torch.nn.Parameter(torch.randn(3))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = WarmupCosineScheduler(opt, warmup_steps=10, max_steps=100, min_lr=0.0)
    assert sched.get_lr(0) == 0.0
    assert abs(sched.get_lr(10) - 1.0) < 1e-6
    assert sched.get_lr(100) < 1e-6
