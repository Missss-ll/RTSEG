
from __future__ import annotations

import torch
import torch.nn.functional as F

def _range_prior(
    range_map: torch.Tensor, alpha: float = 0.02
) -> torch.Tensor:
    r"""Exponential decay of reliability with distance.

    .. math::
        R_{\text{range}}(p) = \exp(-\alpha \cdot r(p))

    Args:
        range_map:  (..., H, W)  Euclidean distance per pixel.
        alpha:      decay coefficient  (larger → faster drop-off).

    r"""Linear ramp: low remission → low reliability, capped at 1.

    .. math::
        R_{\text{rem}}(p) = \min(\text{remission}(p) / \tau,\; 1.0)

    Args:
        remission_map:  (..., H, W)  remission / intensity values.
        tau:            threshold below which reliability scales linearly.

    r"""Penalise depth discontinuities (object boundaries).

    .. math::
        R_{\text{grad}}(p) = \exp(-\beta \cdot |\nabla r(p)|)

    The gradient magnitude is computed via a Sobel-like finite-difference
    kernel applied to the range channel.

    Args:
        range_map:  (..., H, W)  Euclidean distance per pixel.
        beta:       sensitivity to depth edges  (larger → more penalty).

    r"""Compute per-pixel reliability weights from physical priors.

    .. math::
        R(p) = R_{\text{range}} \cdot R_{\text{rem}} \cdot R_{\text{grad}}

    clamped to :math:`[0, 1]`.

    Args:
        range_view:  (B, 5, H, W)  tensor where channels are
                     ``[x, y, z, range, remission]``.
        alpha:       range-decay coefficient.
        tau:         remission threshold.
        beta:        depth-gradient sensitivity.
        smooth_sigma:  std of optional Gaussian smoothing  (0 = off).

    Returns:
        reliability:  (B, H, W)  float tensor in [0, 1].

    Shape:
        Input:  (B, 5, H, W)   →   Output:  (B, H, W)