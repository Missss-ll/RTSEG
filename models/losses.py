"""
Loss functions for RMTSeg training with pseudo-label supervision.

Implements two custom losses from the paper:

1. **ReliabilityLoss** — spatially weighted cross-entropy that down-weights
   unreliable pseudo-label regions based on physically-derived reliability maps.
2. **ReferenceLoss** — distribution-consistency regularisation (MSE / KL) that
   encourages the model output to stay close to a pre-computed reference.

Usage::

    criterion = ReliabilityLoss(label_smoothing=0.1, ignore_index=255)
    loss = criterion(logits, pseudo_labels, reliability_map)

    ref_criterion = ReferenceLoss(mode='mse')
    loss2 = ref_criterion(student_logits, teacher_logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
#  1.  Reliability Loss  —  Spatially Weighted Cross-Entropy
# ======================================================================

class ReliabilityLoss(nn.Module):
    r"""
    Reliability-weighted cross-entropy loss for pseudo-label supervision.

    Given a per-pixel **reliability map** :math:`R \in [0, 1]^{B \times H \times W}`
    (e.g. derived from LiDAR physical priors such as point density, range
    attenuation, or coordinate-transformation uncertainty), this loss scales
    the standard per-pixel cross-entropy by the reliability weight so that
    unreliable pixels contribute less to the gradient::

        L = (1 / ΣR) · Σ_{i,j}  R_{i,j} · CE(pred_{i,j}, label_{i,j})

    This helps the model focus on trustworthy regions during pseudo-label
    training while still receiving weak supervision from uncertain areas.

    Parameters
    ----------
    weight : Tensor | None, default=None
        Optional class-level weight of shape ``(num_classes,)`` passed to
        ``F.cross_entropy``.
    ignore_index : int, default=255
        Label index to ignore (excluded from loss and from the reliability
        normalisation sum).
    label_smoothing : float, default=0.0
        Label smoothing factor (0.0 = none).
    reduction : {'mean', 'sum', 'none'}, default='mean'
        - ``'mean'``   : loss averaged over the *valid* (non-ignored) pixels,
          weighted by reliability.
        - ``'sum'``    : un-normalised sum over all pixels.
        - ``'none'``   : returns a (B, H, W) map of per-pixel weighted CE.
    eps : float, default=1e-8
        Small constant for numerical stability in the normalisation denominator.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
        reduction: str = 'mean',
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.weight = weight          # class weights
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.eps = eps

        # Cache class-weight buffer if provided
        if weight is not None:
            self.register_buffer('class_weight', weight)
        else:
            self.class_weight = None

    def forward(
        self,
        predictions: torch.Tensor,
        pseudo_labels: torch.Tensor,
        reliability_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : (B, C, H, W)
            Raw logits from the segmentation head  (C = num_classes).
        pseudo_labels : (B, H, W)
            LongTensor of pseudo-label indices  (values in [0, C-1] plus
            optionally ``ignore_index``).
        reliability_weights : (B, H, W)
            Spatially-varying reliability map  (values in [0, 1]).

        Returns
        -------
        torch.Tensor
            Scalar loss if ``reduction`` is ``'mean'`` or ``'sum'``;
            (B, H, W) tensor if ``reduction`` is ``'none'``.
        """
        # ---- 1. Per-pixel cross-entropy  (no reduction) --------------------
        ce_map = F.cross_entropy(
            predictions,
            pseudo_labels,
            weight=self.class_weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )                                                    # (B, H, W)

        # ---- 2. Create a validity mask  (exclude ignore_index) --------------
        if self.ignore_index >= 0:
            valid_mask = (pseudo_labels != self.ignore_index).float()  # (B, H, W)
        else:
            valid_mask = torch.ones_like(ce_map)

        # ---- 3. Apply reliability weights + validity mask -------------------
        # Clamp reliability to [0, 1] for safety
        w = reliability_weights.clamp(0.0, 1.0) * valid_mask
        weighted = w * ce_map                                  # (B, H, W)

        # ---- 4. Reduction ---------------------------------------------------
        if self.reduction == 'none':
            return weighted

        if self.reduction == 'sum':
            return weighted.sum()

        # 'mean': normalise by total reliability mass (not pixel count)
        # so that images with many invalid pixels don't artificially inflate
        # the per-image loss.
        w_sum = w.sum() + self.eps
        return weighted.sum() / w_sum


# ======================================================================
#  2.  Reference Loss  —  Distribution-Regularisation Loss
# ======================================================================

class ReferenceLoss(nn.Module):
    r"""
    Reference consistency loss that penalises deviation from a pre-computed
    target distribution.

    Supports two modes:

    * **MSE**  (``mode='mse'``)  — plain :math:`\|x - t\|^2` loss,
      suitable when the reference is a feature map or a soft logit
      distribution (e.g. from a teacher model).
    * **KL divergence**  (``mode='kl'``) — :math:`\mathrm{KL}(p \parallel q)`,
      suitable when both the input and the reference are probability
      distributions (applies log-softmax internally).

    Typical usage in the paper: the student model's intermediate feature
    map or final logits are regularised toward a reference produced by a
    pre-trained teacher, a physics-based model, or a moving-average (EMA)
    copy of the model.

    Parameters
    ----------
    mode : {'mse', 'kl'}, default='mse'
        Distance metric.
    reduction : {'mean', 'sum', 'none'}, default='mean'
        Reduction strategy.
    eps : float, default=1e-8
        Numerical stabiliser for KL mode.
    """

    def __init__(
        self,
        mode: str = 'mse',
        reduction: str = 'mean',
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if mode not in {'mse', 'kl'}:
            raise ValueError(
                f"ReferenceLoss mode must be 'mse' or 'kl', got '{mode}'."
            )
        self.mode = mode
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        reference_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, C, H, W)
            Model predictions  — can be raw logits (for KL mode, log-softmax
            is applied internally) or arbitrary feature tensors (for MSE mode).
        reference_target : (B, C, H, W)
            Reference values of the same shape as ``features``.

        Returns
        -------
        torch.Tensor
            Scalar loss if ``reduction`` is ``'mean'`` or ``'sum'``;
            (B, C, H, W) map if ``reduction`` is ``'none'``.
        """
        if features.shape != reference_target.shape:
            raise ValueError(
                f"Shape mismatch: features {tuple(features.shape)} vs "
                f"reference {tuple(reference_target.shape)}."
            )

        # ---- KL divergence --------------------------------------------------
        if self.mode == 'kl':
            # Log-softmax the model output to get log-probabilities
            log_p = F.log_softmax(features, dim=1)             # (B, C, H, W)
            # Reference must be a valid probability distribution
            q = reference_target.clamp(min=self.eps)
            # KL(p||q) point-wise;  keep channel dim for mean
            kl_map = F.kl_div(
                log_p, q, reduction='none', log_target=False,
            )                                                    # (B, C, H, W)
            kl_map = kl_map.sum(dim=1)                           # (B, H, W)  — KL per pixel

            if self.reduction == 'none':
                return kl_map
            if self.reduction == 'sum':
                return kl_map.sum()
            return kl_map.mean()

        # ---- MSE ------------------------------------------------------------
        # (B, C, H, W)
        se_map = (features - reference_target) ** 2

        if self.reduction == 'none':
            return se_map
        if self.reduction == 'sum':
            return se_map.sum()
        return se_map.mean()


# ======================================================================
#  3.  Combined Reliability + Reference Loss
# ======================================================================

class RMTLoss(nn.Module):
    """
    Convenience wrapper that combines ``ReliabilityLoss`` and ``ReferenceLoss``
    with configurable weights, used as the default training objective for
    RMTSeg when both pseudo-labels and a reference signal are available.

    The total loss is::

        L = λ_rel · L_rel  +  λ_ref · L_ref

    Parameters
    ----------
    num_classes : int
        Number of semantic classes.
    lambda_rel : float, default=1.0
        Weight for the reliability loss.
    lambda_ref : float, default=0.1
        Weight for the reference loss.
    ref_mode : {'mse', 'kl'}, default='mse'
        Distance metric for the reference loss.
    **kwargs
        Forwarded to ``ReliabilityLoss.__init__`` and ``ReferenceLoss.__init__``
        where applicable.
    """

    def __init__(
        self,
        num_classes: int,
        lambda_rel: float = 1.0,
        lambda_ref: float = 0.1,
        ref_mode: str = 'mse',
        **kwargs,
    ) -> None:
        super().__init__()
        self.lambda_rel = lambda_rel
        self.lambda_ref = lambda_ref

        self.rel_loss = ReliabilityLoss(**kwargs)
        self.ref_loss = ReferenceLoss(mode=ref_mode)

    def forward(
        self,
        logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        reliability_weights: torch.Tensor,
        reference_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        logits : (B, C, H, W) — model predictions.
        pseudo_labels : (B, H, W) — pseudo-label indices.
        reliability_weights : (B, H, W) — reliability map ∈ [0, 1].
        reference_logits : (B, C, H, W) — reference / teacher output.

        Returns
        -------
        total_loss : torch.Tensor (scalar)
        loss_dict  : dict with keys ``'rel'``, ``'ref'``, ``'total'``
                     for logging.
        """
        rel = self.rel_loss(logits, pseudo_labels, reliability_weights)
        ref = self.ref_loss(logits, reference_logits)
        total = self.lambda_rel * rel + self.lambda_ref * ref

        return total, {'rel': rel.detach(), 'ref': ref.detach(),
                        'total': total.detach()}
