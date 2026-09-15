"""Model configuration for MEGA-CYBER LLM (MC-LLM).

A single dataclass drives the entire architecture.  Every model size
(100M ... 800B) is expressed as one YAML file loaded through this class.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


def _coerce_number(value: Any) -> Any:
    """Coerce numeric-looking strings (e.g. ``"1e-5"``, ``"123"``) that PyYAML
    fails to resolve into proper numbers."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return value


@dataclass
class ModelConfig:
    # ---- vocabulary / embeddings ----
    vocab_size: int = 32768
    hidden_size: int = 768
    # ---- transformer body ----
    num_layers: int = 12
    num_attention_heads: int = 12
    num_kv_heads: int = 4
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    # ---- positional encoding ----
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    # ---- normalization ----
    rms_norm_eps: float = 1e-5
    # ---- activations / dropout ----
    dropout: float = 0.0
    # ---- ties / flags ----
    tie_word_embeddings: bool = False
    use_bias: bool = False
    # ---- precision ----
    dtype: str = "float32"  # float32 | bfloat16 | float16

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_query_groups(self) -> int:
        """Number of KV groups for grouped-query attention (GQA)."""
        return self.num_attention_heads // self.num_kv_heads

    # ------------------------------------------------------------------
    # parameter counting
    # ------------------------------------------------------------------
    def embedding_parameters(self) -> int:
        return self.vocab_size * self.hidden_size

    def attention_parameters(self) -> int:
        d = self.hidden_size
        hd = self.head_dim
        # q + k + v projections + output projection
        qkv = d * d + (2 * self.num_kv_heads * hd * d)
        out = d * d
        return qkv + out

    def ffn_parameters(self) -> int:
        # SwiGLU: gate + up (2 * d * inter) + down (inter * d)
        return 3 * self.hidden_size * self.intermediate_size

    def normalization_parameters(self) -> int:
        # per-layer: input + post-attention RMSNorm; plus final RMSNorm
        return (2 * self.num_layers + 1) * self.hidden_size

    def output_head_parameters(self) -> int:
        if self.tie_word_embeddings:
            return 0
        return self.vocab_size * self.hidden_size

    def non_embedding_parameters(self) -> int:
        per_layer = self.attention_parameters() + self.ffn_parameters()
        return (self.num_layers * per_layer
                + self.normalization_parameters()
                + self.output_head_parameters())

    def total_parameters(self) -> int:
        total = self.embedding_parameters() + self.non_embedding_parameters()
        if self.tie_word_embeddings:
            total -= self.output_head_parameters()
        return total

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in dataclasses.fields(ModelConfig)}
        return ModelConfig(**{k: _coerce_number(v) for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        model = data.get("model", data) if isinstance(data, dict) else {}
        return cls.from_dict(model)


@dataclass
class TrainingConfig:
    global_batch_size: int = 524288          # total tokens per optimizer step
    micro_batch_size: int = 8                # sequences per forward pass
    sequence_length: int = 2048
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    optimizer: str = "adamw"  # adamw | adam | muon
    scheduler: str = "warmup_cosine"
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 2000
    max_steps: int = 100000
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    dtype: str = "float32"
    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 1000
    seed: int = 42
    use_ema: bool = False
    ema_decay: float = 0.999

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainingConfig":
        d = dict(d)
        if "betas" in d and isinstance(d["betas"], list):
            d["betas"] = tuple(d["betas"])
        known = {f.name for f in dataclasses.fields(TrainingConfig)}
        return TrainingConfig(**{k: _coerce_number(v) for k, v in d.items() if k in known})


@dataclass
class DistributedConfig:
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    sequence_parallel: bool = False
    fsdp: bool = False
    use_gradient_checkpointing: bool = True
    backend: str = "nccl"  # nccl | gloo (gloo is fallback on CPU)

    @property
    def world_size(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DistributedConfig":
        known = {f.name for f in dataclasses.fields(DistributedConfig)}
        return DistributedConfig(**{k: v for k, v in d.items() if k in known})


@dataclass
class RunConfig:
    """Top-level config combining model + training + distributed sections."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    tokenizer_path: str = "tokenizer/tokenizer_config.json"
    data_dir: str = "data/shards"
    out_dir: str = "checkpoints"

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        model = ModelConfig.from_dict(data.get("model", {}))
        training = TrainingConfig.from_dict(data.get("training", {}))
        distributed = DistributedConfig.from_dict(data.get("distributed", {}))
        return cls(
            model=model,
            training=training,
            distributed=distributed,
            tokenizer_path=data.get("tokenizer_path", "tokenizer/tokenizer_config.json"),
            data_dir=data.get("data_dir", "data/shards"),
            out_dir=data.get("out_dir", "checkpoints"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "training": dataclasses.asdict(self.training),
            "distributed": dataclasses.asdict(self.distributed),
            "tokenizer_path": self.tokenizer_path,
            "data_dir": self.data_dir,
            "out_dir": self.out_dir,
        }


def derived_max_steps(config: RunConfig, total_tokens: int) -> int:
    """Back-compat helper: compute steps from token budget if max_steps<=0."""
    if config.training.max_steps > 0:
        return config.training.max_steps
    return max(1, total_tokens // config.training.global_batch_size)
