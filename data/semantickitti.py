
from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

SemanticKITTI_CLASS_NAMES = [
    "unlabeled",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

SemanticKITTI_LEARNING_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
    20: 0,
    21: 0,
    22: 0,
    23: 0,
    24: 0,
    25: 0,
    26: 0,
    27: 0,
    28: 0,
    29: 0,
    30: 0,
    31: 0,
    32: 0,
    33: 0,
    34: 0,
    255: 255,
}
_LEARNING_MAP_VEC = np.vectorize(
    lambda x: SemanticKITTI_LEARNING_MAP.get(x, 255), otypes=[np.uint8]
)

def spherical_projection(
    points: np.ndarray,
    h: int = 64,
    w: int = 2048,
    fov_up: float = 3.0,
    fov_down: float = -25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = np.sqrt(x * x + y * y + z * z)

    yaw = np.arctan2(y, x)
    yaw_norm = (yaw + np.pi) / (2.0 * np.pi)

    pitch = np.arcsin(z / np.clip(r, a_min=1e-6, a_max=None))
    fov_total = fov_up - fov_down
    pitch_norm = (pitch - np.deg2rad(fov_down)) / np.deg2rad(fov_total)

    u = (yaw_norm * w).astype(np.int32)
    v = (pitch_norm * h).astype(np.int32)

    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)

    range_view = np.zeros((h, w, 5), dtype=np.float32)
    label_view = np.full((h, w), 255, dtype=np.uint8)
    depth = np.full((h, w), np.inf, dtype=np.float32)

    remission = points[:, 3]

    for i in range(len(points)):
        vi, ui = v[i], u[i]
        if r[i] < depth[vi, ui]:
            depth[vi, ui] = r[i]
            range_view[vi, ui, 0] = x[i]
            range_view[vi, ui, 1] = y[i]
            range_view[vi, ui, 2] = z[i]
            range_view[vi, ui, 3] = r[i]
            range_view[vi, ui, 4] = remission[i]

    return range_view, label_view

def _spherical_projection_vectorized(
    points: np.ndarray,
    labels: np.ndarray | None,
    h: int = 64,
    w: int = 2048,
    fov_up: float = 3.0,
    fov_down: float = -25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z, remission = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    r = np.sqrt(x * x + y * y + z * z)

    yaw = (np.arctan2(y, x) + np.pi) / (2.0 * np.pi)
    fov_total = fov_up - fov_down
    pitch = (np.arcsin(z / np.clip(r, 1e-6, None)) - np.deg2rad(fov_down)) / np.deg2rad(fov_total)

    u = np.clip((yaw * w).astype(np.int32), 0, w - 1)
    v = np.clip((pitch * h).astype(np.int32), 0, h - 1)

    idx = np.arange(len(points))
    flat_idx = v * w + u

    order = np.lexsort((idx, flat_idx, r))
    _, keep = np.unique(flat_idx[order], return_index=True)
    keep = order[keep]

    range_view = np.zeros((h, w, 5), dtype=np.float32)
    label_view = np.full((h, w), 255, dtype=np.uint8)

    range_view[v[keep], u[keep], 0] = x[keep]
    range_view[v[keep], u[keep], 1] = y[keep]
    range_view[v[keep], u[keep], 2] = z[keep]
    range_view[v[keep], u[keep], 3] = r[keep]
    range_view[v[keep], u[keep], 4] = remission[keep]

    if labels is not None:
        mapped = _LEARNING_MAP_VEC(labels)
        label_view[v[keep], u[keep]] = mapped[keep]

    return range_view, label_view

class SemanticKITTIRangeView(Dataset):

    _train_sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
    _valid_sequences = ["08"]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        h: int = 64,
        w: int = 2048,
        fov_up: float = 3.0,
        fov_down: float = -25.0,
        augment: bool = False,
        flip_prob: float = 0.5,
        noise_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.h = h
        self.w = w
        self.fov_up = fov_up
        self.fov_down = fov_down
        self.augment = augment
        self.flip_prob = flip_prob
        self.noise_std = noise_std

        sequences = (
            self._train_sequences
            if split == "train"
            else self._valid_sequences
        )
        self._samples: list[Tuple[Path, Path]] = []
        for seq in sequences:
            velo_dir = self.data_root / "sequences" / seq / "velodyne"
            label_dir = self.data_root / "sequences" / seq / "labels"
            if not velo_dir.is_dir():
                continue
            for fname in sorted(os.listdir(str(velo_dir))):
                if fname.endswith(".bin"):
                    label_fname = fname.replace(".bin", ".label")
                    label_path = label_dir / label_fname
                    if label_path.is_file():
                        self._samples.append((velo_dir / fname, label_path))

        if len(self._samples) == 0:
            raise FileNotFoundError(
                f"No samples found in {self.data_root}/sequences/{{{','.join(sequences)}}}/"
                f"velodyne/*.bin"
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        bin_path, label_path = self._samples[idx]

        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        raw_labels = np.fromfile(label_path, dtype=np.uint32)
        raw_labels = (raw_labels & 0xFFFF).astype(np.uint8)

        if self.augment:
            points = self._augment(points)

        range_view, label_view = _spherical_projection_vectorized(
            points, raw_labels, self.h, self.w, self.fov_up, self.fov_down
        )

        rv = torch.from_numpy(range_view).permute(2, 0, 1).contiguous()
        lbl = torch.from_numpy(label_view).long()

        return {
            "range_view": rv,
            "labels": lbl,
        }

    def _augment(self, points: np.ndarray) -> np.ndarray:
        p = points.copy()

        if np.random.random() < self.flip_prob:
            p[:, 1] = -p[:, 1]

        if self.noise_std > 0:
            p[:, :3] += np.random.randn(p.shape[0], 3).astype(np.float32) * self.noise_std

        return p

def build_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    h: int = 64,
    w: int = 2048,
    fov_up: float = 3.0,
    fov_down: float = -25.0,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = SemanticKITTIRangeView(
        data_root=data_root,
        split="train",
        h=h, w=w,
        fov_up=fov_up, fov_down=fov_down,
        augment=augment,
    )
    val_ds = SemanticKITTIRangeView(
        data_root=data_root,
        split="valid",
        h=h, w=w,
        fov_up=fov_up, fov_down=fov_down,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
