# RTSEG

Lightweight semantic segmentation with gated visual state space models for LiDAR range-view data.

## Overview

U-Net encoder-decoder where each encoder stage combines global context (cross-scan
selective scan) with local detail (window-based self-attention) through a learned
gated fusion.

## Files

```
├── models/
│   ├── __init__.py
│   ├── rel_mamba.py             # 4-direction cross-scan SSM block
│   ├── local_transformer.py     # Window-based local transformer (W-MSA / SW-MSA)
│   ├── gated_fusion.py          # Gated fusion (basic + bottleneck variants)
│   ├── rmt_seg.py               # Full RMTSeg network
│   └── losses.py                # ReliabilityLoss, ReferenceLoss, RMTLoss
├── data/
│   ├── __init__.py
│   └── semantickitti.py         # SemanticKITTI dataset + spherical projection
├── utils/
│   ├── __init__.py
│   ├── reliability.py           # Physical-prior reliability weights
│   ├── window_select.py         # Difficulty-based adaptive window selection
│   ├── metrics.py               # mIoU / per-class IoU metrics
│   └── checkpoint.py            # Checkpoint save/load
├── configs/
│   └── default.yaml             # Training hyperparameters
├── config.py                    # Configuration management
├── train.py                     # Training script
└── eval.py                      # Evaluation script
```

## Quick start

```python
from models import RMTSeg, RMTLoss

model = RMTSeg(in_channels=5, num_classes=20)
criterion = RMTLoss(num_classes=20, lambda_rel=1.0, lambda_ref=0.1)

logits = model(range_view)
```

## Training

```bash
python train.py --data-root ./data/SemanticKITTI
python eval.py --checkpoint checkpoints/best.pt --data-root ./data/SemanticKITTI
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
- PyYAML
- tqdm
- (optional) `mamba-ssm` for real SSM backend; falls back to Conv1d otherwise

## License

MIT
