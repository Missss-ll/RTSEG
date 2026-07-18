"""
Multi-directional selective scan block following the VMamba cross-scan pattern.
Supports real mamba_ssm backend or a Conv1d fallback when the package is absent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelMambaBlock(nn.Module):
    """
    4-direction cross-scan SSM block with channel-expand gate.

    The 2D feature is scanned along four raster orders (row-major, reversed
    row-major, column-major, reversed column-major), each processed by an
    independent SSM, then merged back and gated element-wise.
    """

    def __init__(self, dim: int, d_state: int = 16, expand: int = 2,
                 use_mamba_ssm: bool = False):
        super().__init__()
        self.dim = dim
        self.expand = expand
        inner_dim = dim * expand

        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, inner_dim * 2)

        self._is_conv1d_fallback = False
        self.ssms = nn.ModuleList([
            self._build_ssm(inner_dim, d_state, use_mamba_ssm)
            for _ in range(4)
        ])

        self.proj_out = nn.Linear(inner_dim, dim)
        self.drop_path = nn.Identity()

    # ------------------------------------------------------------------
    #  SSM backend
    # ------------------------------------------------------------------

    def _build_ssm(self, dim: int, d_state: int, use_mamba_ssm: bool) -> nn.Module:
        if use_mamba_ssm:
            try:
                from mamba_ssm import Mamba
                return Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=1)
            except ImportError:
                pass  # fall through to Conv1d mock
        self._is_conv1d_fallback = True
        return nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)

    # ------------------------------------------------------------------
    #  Cross-scan / cross-merge
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_scan(x: torch.Tensor):
        B, C, H, W = x.shape
        d0 = x.flatten(2).transpose(1, 2)                     # row-major
        d1 = d0.flip(dims=[1])                                 # reversed row-major
        d2 = x.permute(0, 1, 3, 2).flatten(2).transpose(1, 2) # col-major
        d3 = d2.flip(dims=[1])                                 # reversed col-major
        return [d0, d1, d2, d3]

    @staticmethod
    def _cross_merge(scans: list, H: int, W: int) -> torch.Tensor:
        B, _, C = scans[0].shape
        out0 = scans[0].transpose(1, 2).reshape(B, C, H, W)
        out1 = scans[1].flip(dims=[1]).transpose(1, 2).reshape(B, C, H, W)
        out2 = scans[2].transpose(1, 2).reshape(B, C, W, H).permute(0, 1, 3, 2)
        out3 = scans[3].flip(dims=[1]).transpose(1, 2).reshape(B, C, W, H).permute(0, 1, 3, 2)
        return (out0 + out1 + out2 + out3) / 4.0

    def _forward_ssm(self, scan: torch.Tensor, ssm_idx: int) -> torch.Tensor:
        ssm = self.ssms[ssm_idx]
        if self._is_conv1d_fallback:
            s = scan.transpose(1, 2)
            s = F.silu(ssm(s))
            s = s.transpose(1, 2)
            return s
        return ssm(scan)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        shortcut = x

        # LayerNorm (channel-last), then project and split into value / gate
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_flat)
        proj = self.proj_in(x_norm)
        value, gate = proj.chunk(2, dim=-1)
        inner_dim = value.shape[-1]

        # Cross-scan: 2D → 4 × 1D
        value_2d = value.transpose(1, 2).reshape(B, inner_dim, H, W)
        scans = self._cross_scan(value_2d)

        # SSM per direction
        scans_out = [self._forward_ssm(s, i) for i, s in enumerate(scans)]

        # Cross-merge: 4 × 1D → 2D → gated fusion
        merged = self._cross_merge(scans_out, H, W)
        merged = merged.flatten(2).transpose(1, 2)

        gate = F.silu(gate)
        output = merged * gate

        # Project back and residual
        output = self.proj_out(output)
        output = output.reshape(B, H, W, C).permute(0, 3, 1, 2)
        output = shortcut + self.drop_path(output)
        return output
