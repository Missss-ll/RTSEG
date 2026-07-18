"""
GatedFusion: Learnable Gating Module for Multi-Branch Feature Fusion.

Fuses a global-context feature map (from the VMamba branch) with a
local-detail feature map (from the window-transformer branch) via a
content-aware sigmoid gate.

Reference pattern:
    "Gated Fusion Network for Semantic Segmentation" and similar works
    that employ learned gating for bi-modal / multi-branch fusion.

Architecture overview::

    feat_global (B, C, H, W)     feat_local (B, C, H, W)
            │                              │
            └─────────── concat ───────────┘
                         │
                   Conv2d(2C → gate_c, 1×1)
                         │
                      Sigmoid
                         │
                  gate ∈ (B, gate_c, H, W)
                         │
            ┌────────────┴────────────┐
            │                         │
    feat_global ⊙ gate       feat_local ⊙ (1 - gate)
            │                         │
            └───────────  +  ─────────┘
                         │
                 output (B, C, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """
    Content-aware gated fusion of two feature branches.

    Concatenates the two input feature maps, produces a sigmoid-gated weight
    map via a 1×1 convolution, then blends the two branches via element-wise
    weighted summation.

    Parameters
    ----------
    dim : int
        Number of input channels  (both branches must have the same ``dim``).
    gate_channels : int, default=1
        Dimensionality of the gating map.
        - ``1``  → spatial-only gate  (B, 1, H, W) — one scalar weight per
          spatial location shared across all channels.
        - ``dim`` → channel-wise gate (B, C, H, W) — each channel at each
          spatial position is independently weighted.
    bias : bool, default=False
        If True, the conv layer includes a bias term.
    gate_init_bias : float, default=0.0
        Initial bias value for the conv layer.  A positive value biases the
        gate toward 1 (favoring the global branch early in training); a
        negative value favors the local branch.  When ``bias=True``, the
        conv bias is initialised to this constant.
    """

    def __init__(
        self,
        dim: int,
        gate_channels: int = 1,
        bias: bool = False,
        gate_init_bias: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.gate_channels = gate_channels

        # ---- Gate generator: concat → 1×1 conv → sigmoid -------------------
        self.gate_conv = nn.Conv2d(
            in_channels=dim * 2,
            out_channels=gate_channels,
            kernel_size=1,
            bias=bias,
        )

        # Initialise bias to control early-training balance
        if bias:
            nn.init.constant_(self.gate_conv.bias, gate_init_bias)

        # Initialise weights with a small variance for stable early training
        nn.init.normal_(self.gate_conv.weight, mean=0.0, std=0.001)

    def forward(
        self,
        feat_global: torch.Tensor,
        feat_local: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        feat_global : (B, C, H, W)
            Feature map from the global-context branch  (e.g. RelMambaBlock).
        feat_local : (B, C, H, W)
            Feature map from the local-detail branch  (e.g. LocalTransformerBlock).
            Must have the same spatial resolution as ``feat_global``.

        Returns
        -------
        (B, C, H, W)
            Fused feature map at the same resolution.
        """
        # ---- 1. Concatenate along channel dimension -------------------------
        # (B, 2C, H, W)
        concat = torch.cat([feat_global, feat_local], dim=1)

        # ---- 2. Gate prediction: 1×1 conv → sigmoid ------------------------
        # (B, gate_c, H, W)
        gate = self.gate_conv(concat)
        gate = torch.sigmoid(gate)

        # ---- 3. Gated fusion: weighted sum ----------------------------------
        # If gate_channels != C, broadcast along the channel dimension
        out = gate * feat_global + (1.0 - gate) * feat_local

        return out


class GatedFusionV2(nn.Module):
    """
    Extended gate with a small two-layer bottleneck for richer context.

    Compared to ``GatedFusion``, this variant adds a squeeze-expand path
    (reduction → SiLU → expansion) before the sigmoid, giving the gate
    access to a slightly larger receptive field without a heavy parameter
    overhead.

    Parameters
    ----------
    dim : int
        Channel dimension of both input branches.
    reduction : int, default=4
        Bottleneck reduction ratio  (hidden_dim = dim // reduction).
    gate_channels : int, default=1
        Output gate map channels.
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 4,
        gate_channels: int = 1,
    ):
        super().__init__()
        hidden = max(dim // reduction, 8)

        self.gate_net = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, gate_channels, kernel_size=1, bias=True),
        )

        # Initialise the final conv bias to 0 (balanced gate at start)
        nn.init.constant_(self.gate_net[-1].bias, 0.0)
        # Small initial weights → gate ≈ 0.5 early on
        nn.init.normal_(self.gate_net[-1].weight, mean=0.0, std=0.001)

    def forward(
        self,
        feat_global: torch.Tensor,
        feat_local: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        feat_global : (B, C, H, W)
        feat_local  : (B, C, H, W)

        Returns
        -------
        (B, C, H, W)
        """
        concat = torch.cat([feat_global, feat_local], dim=1)     # (B, 2C, H, W)
        gate = self.gate_net(concat)                              # (B, gate_c, H, W)
        gate = torch.sigmoid(gate)
        return gate * feat_global + (1.0 - gate) * feat_local
