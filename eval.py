

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from config import Config, load_config
from data.semantickitti import build_dataloaders, SemanticKITTI_CLASS_NAMES
from models import RMTSeg, rmt_seg_tiny, rmt_seg_small, rmt_seg_base
from utils.metrics import SegmentationMetrics
from utils.checkpoint import load_checkpoint

VARIANT_MAP = {
    "rmt_seg_tiny": rmt_seg_tiny,
    "rmt_seg_small": rmt_seg_small,
    "rmt_seg_base": rmt_seg_base,
}

def build_model_from_checkpoint(
    ckpt_path: str,
    cfg: Config,
    device: torch.device,
) -> RMTSeg:
    variant = cfg.model.variant
    if variant in VARIANT_MAP:
        model = VARIANT_MAP[variant](**cfg.model.to_dict())
    else:
        model = RMTSeg(**cfg.model.to_dict())
    model = model.to(device)

    load_checkpoint(model, ckpt_path, map_location=str(device))
    return model

@torch.no_grad()
def evaluate(
    model: RMTSeg,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict:
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=255)

    pbar = tqdm(loader, desc="Evaluating")
    for batch in pbar:
        rv = batch["range_view"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(rv)
        preds = logits.argmax(dim=1)
        metrics.update(preds, labels)

    return metrics.compute()

def main() -> None:
    parser = argparse.ArgumentParser(description="RTSEG Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pt file).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (uses same as training).")
    parser.add_argument("--data-root", type=str, default=None,
                        help="SemanticKITTI dataset root.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda / cpu).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size.")
    parser.add_argument("--output", type=str, default=None,
                        help="Save metrics to JSON file.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config(args.config)
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.batch_size:
        cfg.data.batch_size = args.batch_size

    _, val_loader = build_dataloaders(
        data_root=cfg.data.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        h=cfg.data.h,
        w=cfg.data.w,
        fov_up=cfg.data.fov_up,
        fov_down=cfg.data.fov_down,
        augment=False,
    )
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = build_model_from_checkpoint(args.checkpoint, cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.2f} M")

    results = evaluate(model, val_loader, cfg.model.num_classes, device)

    print("\n" + "=" * 60)
    print(f"  mIoU:       {results['mIoU'] * 100:.2f}%")
    print(f"  Pixel Acc:  {results['pixel_acc'] * 100:.2f}%")
    print(f"  Mean Acc:   {results['mean_acc'] * 100:.2f}%")
    print("-" * 60)
    print("  Per-class IoU:")

    class_names = SemanticKITTI_CLASS_NAMES
    for i, iou in enumerate(results["per_class_IoU"]):
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        marker = "  ←" if iou < 0.1 else ""
        print(f"    {i:2d} {name:<20s} {iou * 100:6.2f}%{marker}")

    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()