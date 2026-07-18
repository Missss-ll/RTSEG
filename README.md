# RTSEG

Lightweight semantic segmentation with gated visual state space models for LiDAR range-view data.

## Overview

U-Net encoder-decoder where each encoder stage combines global context (cross-scan
selective scan) with local detail (window-based self-attention) through a learned
gated fusion.

## Files

```
models/
├── __init__.py
├── rel_mamba.py             # 4-direction cross-scan SSM block
├── local_transformer.py     # Window-based local transformer (W-MSA / SW-MSA)
├── gated_fusion.py          # Gated fusion (basic + bottleneck variants)
├── rmt_seg.py               # Full RMTSeg network
└── losses.py                # ReliabilityLoss, ReferenceLoss, RMTLoss
```

## Quick start

```python
from models import RMTSeg, RMTLoss

model = RMTSeg(in_channels=5, num_classes=20)
criterion = RMTLoss(num_classes=20, lambda_rel=1.0, lambda_ref=0.1)

# Input:  range-view (B, 5, H, W)
# Output: logits    (B, 20, H, W)
logits = model(range_view)
```

## Variants

| Variant | embed_dim | stages | blocks |
|---------|-----------|--------|--------|
| `rmt_seg_tiny()`  | 32 | 3 | [1, 1, 1] |
| `rmt_seg_small()` | 64 | 3 | [1, 1, 1] |
| `rmt_seg_base()`  | 64 | 4 | [1, 1, 2, 1] |

## Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- (optional) `mamba-ssm` for real SSM backend; falls back to Conv1d otherwise

## License

MIT
