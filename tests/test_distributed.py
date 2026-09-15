"""Distributed/tensor-parallel math tests (run in single process)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_column_parallel_linear_math():
    """Verify column-parallel split equals a monolithic linear (simulated by
    sharding manually without an actual process group)."""
    from training.distributed import ColumnParallelLinear, RowParallelLinear

    torch.manual_seed(0)
    in_f, out_f = 8, 4
    x = torch.randn(2, in_f)

    ref = torch.nn.Linear(in_f, out_f)
    ref_out = ref(x)

    # manual 2-way column split
    half = out_f // 2
    w1 = ref.weight[:half]
    w2 = ref.weight[half:]
    b1 = ref.bias[:half]
    b2 = ref.bias[half:]
    out1 = torch.nn.functional.linear(x, w1, b1)
    out2 = torch.nn.functional.linear(x, w2, b2)
    combined = torch.cat([out1, out2], dim=-1)
    assert torch.allclose(combined, ref_out, atol=1e-5)


def test_row_parallel_linear_math():
    torch.manual_seed(1)
    in_f, out_f = 8, 4
    x = torch.randn(2, in_f)
    ref = torch.nn.Linear(in_f, out_f)
    ref_out = ref(x)

    # manual 2-way row split: sum of partial matmuls
    half = in_f // 2
    x1, x2 = x[:, :half], x[:, half:]
    w1, w2 = ref.weight[:, :half], ref.weight[:, half:]
    partial = torch.nn.functional.linear(x1, w1) + torch.nn.functional.linear(x2, w2)
    partial = partial + ref.bias
    assert torch.allclose(partial, ref_out, atol=1e-5)


def test_rope_is_position_sensitive():
    from model.attention import precompute_rope, apply_rope
    cos, sin = precompute_rope(16, 32)
    x = torch.randn(1, 32, 16)
    pos0 = apply_rope(x, cos[0], sin[0])
    pos5 = apply_rope(x, cos[5], sin[5])
    assert not torch.allclose(pos0, pos5)
