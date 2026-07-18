"""
Content-aware gated fusion for blending global (SSM) and local (attention) branches.
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Fuse two feature maps via a learned sigmoid gate (1x1 conv over concat).

    gate_channels=1 gives a spatial-only gate; gate_channels=C gives
    per-channel gating.
    """

    def __init__(self, dim: int, gate_channels: int = 1, bias: bool = False,
                 gate_init_bias: float = 0.0):
        super().__init__()
        self.dim = dim
        self.gate_channels = gate_channels

        self.gate_conv = nn.Conv2d(dim * 2, gate_channels, kernel_size=1, bias=bias)
        if bias:
            nn.init.constant_(self.gate_conv.bias, gate_init_bias)
        nn.init.normal_(self.gate_conv.weight, mean=0.0, std=0.001)

    def forward(self, feat_global: torch.Tensor,
                feat_local: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([feat_global, feat_local], dim=1)
        gate = torch.sigmoid(self.gate_conv(concat))
        return gate * feat_global + (1.0 - gate) * feat_local


class GatedFusionV2(nn.Module):
    """Gated fusion with a squeeze-expand bottleneck for richer gate context."""

    def __init__(self, dim: int, reduction: int = 4, gate_channels: int = 1):
        super().__init__()
        hidden = max(dim // reduction, 8)

        self.gate_net = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, gate_channels, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.gate_net[-1].bias, 0.0)
        nn.init.normal_(self.gate_net[-1].weight, mean=0.0, std=0.001)

    def forward(self, feat_global: torch.Tensor,
                feat_local: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([feat_global, feat_local], dim=1)
        gate = torch.sigmoid(self.gate_net(concat))
        return gate * feat_global + (1.0 - gate) * feat_local
