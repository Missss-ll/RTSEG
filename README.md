# RTSEG

**Lightweight Semantic Segmentation Algorithm Based on Gated Visual State Space Models**

A PyTorch implementation of a U-Net-style encoder-decoder network for LiDAR point-cloud semantic segmentation on 2D range-view projections.

## Overview

Each encoder stage combines:

- **RelMambaBlock** — VMamba-style 4-directional cross-scan selective scan for global context.
- **LocalTransformerBlock** — Swin-style window-based self-attention for local detail.
- **GatedFusion** — content-aware sigmoid gate to blend the two branches.

## Structure

```
models/
├── __init__.py              # Package exports
├── rel_mamba.py             # Multi-directional SSM block
├── local_transformer.py     # Window-based local transformer
├── gated_fusion.py          # Gated fusion modules
├── rmt_seg.py               # Full RMTSeg network
└── losses.py                # ReliabilityLoss + ReferenceLoss + RMTLoss
```

## Quick Start

```python
from models import RMTSeg, RMTLoss

model = RMTSeg(in_channels=5, num_classes=20)
criterion = RMTLoss(num_classes=20, lambda_rel=1.0, lambda_ref=0.1)

# Input: range-view projection  (B, 5, H, W)
# Output: per-pixel logits     (B, 20, H, W)
logits = model(range_view_tensor)
```

## Model Variants

| Variant | `embed_dim` | Stages | Blocks |
|---------|------------|--------|--------|
| `rmt_seg_tiny()`  | 32  | 3 | [1, 1, 1] |
| `rmt_seg_small()` | 64  | 3 | [1, 1, 1] |
| `rmt_seg_base()`  | 64  | 4 | [1, 1, 2, 1] |

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- (Optional) `mamba-ssm` for the real SSM backend (falls back to Conv1d mock)

## License

MIT
