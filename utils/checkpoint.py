
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
    epoch: int,
    best_metric: float,
    path: str | Path,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt: dict = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    ckpt["rng_states"] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        ckpt["rng_states"]["cuda"] = torch.cuda.get_rng_state_all()

    if extra is not None:
        ckpt["extra"] = extra

    torch.save(ckpt, str(path))

def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    map_location: str = "cpu",
    restore_rng: bool = False,
) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if restore_rng and "rng_states" in ckpt:
        rng = ckpt["rng_states"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

    return {
        "epoch": ckpt.get("epoch", 0),
        "best_metric": ckpt.get("best_metric", 0.0),
        "extra": ckpt.get("extra", {}),
    }