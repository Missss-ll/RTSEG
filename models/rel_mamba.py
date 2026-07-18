"""
RelMambaBlock: VMamba-style Multi-Directional Selective Scan Module.

Implements the Cross-Scan Selective Scan (SS2D / Cross-Scan) mechanism
proposed in "VMamba: Visual State Space Model" (Liu et al., 2024).

Core idea: a 2D feature map is unfolded into four 1D sequences along
different scan directions (row-major, column-major, and their reversals).
Each sequence is processed independently by a State Space Model (SSM), and
the four outputs are fused back into a 2D map via averaging.

For the SSM backend, this module supports both:
  - A Conv1d-based mock (always available, no external dependency).
  - The official ``mamba_ssm`` package  (optional, set ``use_mamba_ssm=True``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelMambaBlock(nn.Module):
    """
    VMamba-style block with 4-directional cross-scan and gated SSM.

    Pipeline overview::

        x (B, C, H, W)
        │
        ├─ LayerNorm ─► Linear( C → 2·E·C ) ─► chunk ─┬─► value
        │                                               └─► gate
        │ value: cross_scan ─► 4 × SSM ─► cross_merge ─► merged
        │ gate:   SiLU ─────────────────────────────────► gated
        │
        └─ merged ⊙ gated ─► Linear( E·C → C ) ─► + ─► output (B, C, H, W)

    Parameters
    ----------
    dim : int
        Input / output channel dimension.
    d_state : int, default=16
        SSM state-expansion factor (used only with the real mamba_ssm backend).
    expand : int, default=2
        Channel expansion ratio inside the block  (inner_dim = dim * expand).
    use_mamba_ssm : bool, default=False
        If True, attempts to import ``mamba_ssm.Mamba``.  Falls back to a
        Conv1d-based mock when the package is not available.
    """

    def __init__(self, dim: int, d_state: int = 16, expand: int = 2,
                 use_mamba_ssm: bool = False):
        super().__init__()
        self.dim = dim
        self.expand = expand
        inner_dim = dim * expand

        # -- Normalisation & projections ---------------------------------------
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, inner_dim * 2)  # value + gate branches

        # -- SSM processors (one independent SSM per scan direction) -----------
        self._is_conv1d_fallback = False
        self.ssms = nn.ModuleList([
            self._build_ssm(inner_dim, d_state, use_mamba_ssm)
            for _ in range(4)
        ])

        self.proj_out = nn.Linear(inner_dim, dim)

        # DropPath placeholder — replace with e.g. timm's DropPath when needed
        self.drop_path = nn.Identity()

    # ------------------------------------------------------------------
    # SSM backend construction
    # ------------------------------------------------------------------

    def _build_ssm(self, dim: int, d_state: int, use_mamba_ssm: bool) -> nn.Module:
        """
        Return an SSM processor for one scan direction.

        When ``use_mamba_ssm=True`` and the package is installed, returns a
        real Mamba block.  Otherwise returns a depth-wise Conv1d as a
        simplified state-space proxy.
        """
        if use_mamba_ssm:
            try:
                from mamba_ssm import Mamba  # type: ignore[import-untyped]
                # Mamba operates on (B, L, C) — convenient for scan sequences
                return Mamba(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=4,
                    expand=1,          # expansion already handled by proj_in
                )
            except ImportError:
                print("[RelMambaBlock] mamba_ssm not found; "
                      "falling back to Conv1d mock.")
        # Mock: depth-wise Conv1d ≈ diagonal state-space dynamics
        self._is_conv1d_fallback = True
        return nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)

    # ------------------------------------------------------------------
    # Cross-Scan & Cross-Merge  (static methods — pure tensor ops)
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_scan(x: torch.Tensor):
        """
        Unfold a 2D feature map into four 1D sequences by scanning along
        different traversal orders.

        Directions
        ----------
        d0 : row-major          — top-left → bottom-right  (standard raster)
        d1 : reversed row-major — bottom-right → top-left
        d2 : column-major       — top-left → bottom-right  (transposed raster)
        d3 : reversed col-major — bottom-right → top-left

        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        list of four (B, H*W, C) tensors
        """
        B, C, H, W = x.shape
        # d0 — standard row-major flatten
        d0 = x.flatten(2).transpose(1, 2)                    # (B, H*W, C)
        # d1 — reverse of d0
        d1 = d0.flip(dims=[1])                                # (B, H*W, C)
        # d2 — column-major flatten: swap H↔W, then flatten
        d2 = x.permute(0, 1, 3, 2).flatten(2).transpose(1, 2)  # (B, W*H, C)
        # d3 — reverse of d2
        d3 = d2.flip(dims=[1])                                # (B, W*H, C)
        return [d0, d1, d2, d3]

    @staticmethod
    def _cross_merge(scans: list, H: int, W: int) -> torch.Tensor:
        """
        Fold four 1D directional sequences back into a single 2D feature map.

        Each sequence is un-flattened according to its scan order, and the
        resulting four 2D maps are averaged (mean fusion).

        Parameters
        ----------
        scans : list of four (B, H*W, C) tensors
            Must follow the same order as returned by ``_cross_scan``.
        H, W : int
            Target spatial resolution.

        Returns
        -------
        (B, C, H, W) merged feature map
        """
        B, _, C = scans[0].shape
        # d0 — row-major → reshape directly
        out0 = scans[0].transpose(1, 2).reshape(B, C, H, W)
        # d1 — reversed row-major → un-reverse → reshape
        out1 = scans[1].flip(dims=[1]).transpose(1, 2).reshape(B, C, H, W)
        # d2 — column-major → reshape as (W, H) → permute to (H, W)
        out2 = scans[2].transpose(1, 2).reshape(B, C, W, H).permute(0, 1, 3, 2)
        # d3 — reversed column-major → un-reverse → reshape → permute
        out3 = scans[3].flip(dims=[1]).transpose(1, 2).reshape(B, C, W, H).permute(0, 1, 3, 2)
        # Mean fusion across four directions
        return (out0 + out1 + out2 + out3) / 4.0

    # ------------------------------------------------------------------
    # SSM forward helper
    # ------------------------------------------------------------------

    def _forward_ssm(self, scan: torch.Tensor, ssm_idx: int) -> torch.Tensor:
        """
        Process one scan sequence through its designated SSM.

        Handles the layout difference:
          - Real Mamba : ``(B, L, C) → (B, L, C)``
          - Conv1d mock: ``(B, C, L) → (B, C, L)``

        Parameters
        ----------
        scan : (B, L, C)
        ssm_idx : int in [0, 3]

        Returns
        -------
        (B, L, C) processed sequence
        """
        ssm = self.ssms[ssm_idx]
        if self._is_conv1d_fallback:
            # Conv1d mock expects (B, C, L)
            s = scan.transpose(1, 2)          # (B, C, L)
            s = F.silu(ssm(s))                # depth-wise conv + non-linearity
            s = s.transpose(1, 2)             # (B, L, C)
            return s
        else:
            # Real Mamba expects (B, L, C)
            return ssm(scan)

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)
            Input feature map (e.g. a range-view distance image).

        Returns
        -------
        (B, C, H, W)
            Output feature map with the same spatial resolution.
        """
        B, C, H, W = x.shape
        shortcut = x

        # ---- 1. LayerNorm  (applied channel-last) ---------------------------
        # (B, C, H, W) → (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_flat)                               # (B, H*W, C)

        # ---- 2. Channel expansion & gate split ------------------------------
        proj = self.proj_in(x_norm)                              # (B, H*W, 2·inner)
        value, gate = proj.chunk(2, dim=-1)                      # (B, H*W, inner) × 2
        inner_dim = value.shape[-1]

        # ---- 3. Cross-scan: 2D → 4 × 1D sequences --------------------------
        value_2d = value.transpose(1, 2).reshape(B, inner_dim, H, W)  # 2D
        scans = self._cross_scan(value_2d)                       # list of 4

        # ---- 4. SSM processing (independent SSM per direction) --------------
        scans_out = [
            self._forward_ssm(s, i) for i, s in enumerate(scans)
        ]                                                        # list of 4

        # ---- 5. Cross-merge: 4 × 1D → 2D -----------------------------------
        merged = self._cross_merge(scans_out, H, W)              # (B, inner, H, W)
        merged = merged.flatten(2).transpose(1, 2)               # (B, H*W, inner)

        # ---- 6. Gating ------------------------------------------------------
        gate = F.silu(gate)                                      # (B, H*W, inner)
        output = merged * gate                                   # element-wise gate

        # ---- 7. Output projection & residual --------------------------------
        output = self.proj_out(output)                           # (B, H*W, C)
        output = output.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        output = shortcut + self.drop_path(output)

        return output
