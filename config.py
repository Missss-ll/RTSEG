
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

@dataclass
class ModelConfig:
    in_channels: int = 5
    num_classes: int = 20
    embed_dim: int = 64
    num_stages: int = 3
    num_blocks: list[int] = field(default_factory=lambda: [1, 1, 1])
    window_size: int = 7
    num_heads: int = 8
    mamba_d_state: int = 16
    mamba_expand: int = 2
    mlp_ratio: float = 4.0
    use_mamba_ssm: bool = False
    gate_v2: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    drop_path_rate: float = 0.1
    head_dropout: float = 0.1
    stem_stride: int = 1
    stem_skip: bool = True
    adaptive_window: bool = False
    window_keep_ratio: float = 0.7
    variant: str = "rmt_seg_small"

    def to_dict(self) -> dict[str, Any]:
        d = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name
            not in {
                "variant",
                "adaptive_window",
                "window_keep_ratio",
            }
        }
        if isinstance(d["num_blocks"], int):
            d["num_blocks"] = [d["num_blocks"]]
        return d

@dataclass
class DataConfig:
    data_root: str = "./data/SemanticKITTI"
    batch_size: int = 8
    num_workers: int = 4
    h: int = 64
    w: int = 2048
    fov_up: float = 3.0
    fov_down: float = -25.0
    augment: bool = True
    flip_prob: float = 0.5
    noise_std: float = 0.02

@dataclass
class ReliabilityConfig:
    alpha: float = 0.02
    tau: float = 0.1
    beta: float = 0.5
    smooth_sigma: float = 1.0

@dataclass
class LossConfig:
    lambda_rel: float = 1.0
    lambda_ref: float = 0.0
    ref_mode: str = "mse"
    label_smoothing: float = 0.0
    class_weight: str | None = None

@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0001
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1.0e-6
    step_size: int = 30
    step_gamma: float = 0.1
    plateau_patience: int = 10
    grad_clip: float = 1.0
    amp: bool = True
    ema_decay: float = 0.0
    log_interval: int = 50
    val_interval: int = 1

@dataclass
class CheckpointConfig:
    save_dir: str = "./checkpoints"
    save_best: bool = True
    resume: str | None = None

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        if yaml is None:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install it with: pip install pyyaml"
            )
        with open(path, "r", encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f) or {}

        def _populate(section: str, dc: Any) -> Any:
            data = raw.get(section, {})
            if data is None:
                data = {}
            field_names = {f.name for f in fields(dc)}
            return dc(**{k: v for k, v in data.items() if k in field_names})

        return cls(
            model=_populate("model", ModelConfig),
            data=_populate("data", DataConfig),
            reliability=_populate("reliability", ReliabilityConfig),
            loss=_populate("loss", LossConfig),
            train=_populate("train", TrainConfig),
            checkpoint=_populate("checkpoint", CheckpointConfig),
        )

    def to_yaml(self, path: str | Path) -> None:
        if yaml is None:
            raise ImportError(
                "PyYAML is required. Install it with: pip install pyyaml"
            )
        out: dict[str, Any] = {}
        for section_name, dc in [
            ("model", self.model),
            ("data", self.data),
            ("reliability", self.reliability),
            ("loss", self.loss),
            ("train", self.train),
            ("checkpoint", self.checkpoint),
        ]:
            out[section_name] = {
                f.name: getattr(dc, f.name) for f in fields(dc)
            }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, default_flow_style=False, sort_keys=False)

