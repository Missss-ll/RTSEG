
from __future__ import annotations

import torch

class SegmentationMetrics:

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self._cm = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64
        )

    @torch.no_grad()
    def update(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        pred_flat = predictions.reshape(-1).long()
        target_flat = targets.reshape(-1).long()

        if self.ignore_index >= 0:
            valid = target_flat != self.ignore_index
            pred_flat = pred_flat[valid]
            target_flat = target_flat[valid]

        pred_flat = pred_flat.clamp(0, self.num_classes - 1)
        target_flat = target_flat.clamp(0, self.num_classes - 1)

        idx = target_flat * self.num_classes + pred_flat
        cm_batch = torch.bincount(
            idx, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)

        self._cm += cm_batch.cpu()

    def compute(self) -> dict[str, float]:
        cm = self._cm.float()
        intersection = cm.diag()
        union = cm.sum(dim=0) + cm.sum(dim=1) - intersection

        iou = intersection / union.clamp(min=1e-8)
        iou[union == 0] = float("nan")

        present = union > 0
        mIoU = iou[present].mean().item() if present.any() else 0.0

        pixel_acc = (
            intersection.sum() / cm.sum().clamp(min=1e-8)
        ).item()

        row_sum = cm.sum(dim=1)
        per_class_acc = intersection / row_sum.clamp(min=1e-8)
        mean_acc = per_class_acc[present].mean().item() if present.any() else 0.0

        return {
            "mIoU": mIoU,
            "per_class_IoU": [float(v) for v in iou.tolist()],
            "pixel_acc": pixel_acc,
            "mean_acc": mean_acc,
        }

    def summary(self) -> str:
        res = self.compute()
        lines = [
            f"mIoU:       {res['mIoU'] * 100:.2f}%",
            f"pixel_acc:  {res['pixel_acc'] * 100:.2f}%",
            f"mean_acc:   {res['mean_acc'] * 100:.2f}%",
        ]
        return "\n".join(lines)

