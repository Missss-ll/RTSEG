"""
Training losses: reliability-weighted cross-entropy and reference consistency (MSE / KL).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityLoss(nn.Module):
    """Per-pixel cross-entropy weighted by a spatial reliability map R ∈ [0,1].

    L = (Σ R * CE(pred, label)) / (Σ R)

    Pixels with low reliability (e.g. sparse regions in range-view) are
    down-weighted so they contribute less to the gradient.
    """

    def __init__(self, weight: torch.Tensor | None = None,
                 ignore_index: int = 255, label_smoothing: float = 0.0,
                 reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.eps = eps
        if weight is not None:
            self.register_buffer('class_weight', weight)
        else:
            self.class_weight = None

    def forward(self, predictions: torch.Tensor, pseudo_labels: torch.Tensor,
                reliability_weights: torch.Tensor) -> torch.Tensor:
        # Per-pixel CE, no reduction
        ce_map = F.cross_entropy(
            predictions, pseudo_labels, weight=self.class_weight,
            ignore_index=self.ignore_index, label_smoothing=self.label_smoothing,
            reduction='none',
        )
        # Mask out ignored pixels
        if self.ignore_index >= 0:
            valid_mask = (pseudo_labels != self.ignore_index).float()
        else:
            valid_mask = torch.ones_like(ce_map)

        w = reliability_weights.clamp(0.0, 1.0) * valid_mask
        weighted = w * ce_map

        if self.reduction == 'none':
            return weighted
        if self.reduction == 'sum':
            return weighted.sum()
        return weighted.sum() / (w.sum() + self.eps)


class ReferenceLoss(nn.Module):
    """MSE or KL divergence between model output and a reference target.

    MSE is suitable for feature-map regularisation; KL is for probability
    distributions (applies log-softmax internally).
    """

    def __init__(self, mode: str = 'mse', reduction: str = 'mean',
                 eps: float = 1e-8):
        super().__init__()
        if mode not in {'mse', 'kl'}:
            raise ValueError(f"mode must be 'mse' or 'kl', got '{mode}'.")
        self.mode = mode
        self.reduction = reduction
        self.eps = eps

    def forward(self, features: torch.Tensor,
                reference_target: torch.Tensor) -> torch.Tensor:
        if features.shape != reference_target.shape:
            raise ValueError(
                f"Shape mismatch: {tuple(features.shape)} vs "
                f"{tuple(reference_target.shape)}."
            )

        if self.mode == 'kl':
            log_p = F.log_softmax(features, dim=1)
            q = reference_target.clamp(min=self.eps)
            kl_map = F.kl_div(log_p, q, reduction='none', log_target=False)
            kl_map = kl_map.sum(dim=1)
            if self.reduction == 'none':
                return kl_map
            if self.reduction == 'sum':
                return kl_map.sum()
            return kl_map.mean()

        # MSE
        se_map = (features - reference_target) ** 2
        if self.reduction == 'none':
            return se_map
        if self.reduction == 'sum':
            return se_map.sum()
        return se_map.mean()


class RMTLoss(nn.Module):
    """Combined loss: λ_rel * ReliabilityLoss + λ_ref * ReferenceLoss."""

    def __init__(self, num_classes: int, lambda_rel: float = 1.0,
                 lambda_ref: float = 0.1, ref_mode: str = 'mse', **kwargs):
        super().__init__()
        self.lambda_rel = lambda_rel
        self.lambda_ref = lambda_ref
        self.rel_loss = ReliabilityLoss(**kwargs)
        self.ref_loss = ReferenceLoss(mode=ref_mode)

    def forward(self, logits: torch.Tensor, pseudo_labels: torch.Tensor,
                reliability_weights: torch.Tensor,
                reference_logits: torch.Tensor):
        rel = self.rel_loss(logits, pseudo_labels, reliability_weights)
        ref = self.ref_loss(logits, reference_logits)
        total = self.lambda_rel * rel + self.lambda_ref * ref
        return total, {'rel': rel.detach(), 'ref': ref.detach(),
                       'total': total.detach()}
