"""Config tests: YAML loading and parameter calculations."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import RunConfig, ModelConfig


def test_all_configs_load():
    cfg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "configs")
    for name in ["100m", "300m", "1b", "3b", "7b", "13b", "70b", "405b", "800b"]:
        cfg = RunConfig.from_yaml(os.path.join(cfg_dir, f"{name}.yaml"))
        assert cfg.model.total_parameters() > 0


def test_config_sizes_reasonable():
    cfg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "configs")
    expected = {
        "100m": (40e6, 150e6),
        "300m": (250e6, 500e6),
        "1b": (0.8e9, 1.3e9),
        "3b": (2.5e9, 4e9),
        "7b": (6e9, 9e9),
        "13b": (10e9, 14e9),
        "70b": (60e9, 80e9),
        "405b": (350e9, 460e9),
        "800b": (700e9, 900e9),
    }
    for name, (lo, hi) in expected.items():
        cfg = RunConfig.from_yaml(os.path.join(cfg_dir, f"{name}.yaml"))
        n = cfg.model.total_parameters()
        assert lo <= n <= hi, f"{name}: {n} not in [{lo}, {hi}]"


def test_head_dim_divisible():
    cfg = RunConfig.from_yaml(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "7b.yaml"))
    assert cfg.model.hidden_size % cfg.model.num_attention_heads == 0
    assert cfg.model.num_attention_heads % cfg.model.num_kv_heads == 0
