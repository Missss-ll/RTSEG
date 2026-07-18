"""
RMTSeg: Range-view Mamba-Transformer Semantic Segmentation Network.

A lightweight U-Net-style encoder-decoder architecture for LiDAR point-cloud
semantic segmentation on 2D range-view projections.

Each encoder stage runs a RelMambaBlock (global SSM context) in parallel with a
LocalTransformerBlock (window-based local attention), fuses the two via a
learned GatedFusion module, then spatially downsamples.

Reference:
    "Lightweight Semantic Segmentation Algorithm Based on Gated Visual State
     Space Models" — the network combines VMamba-style cross-scan selective
     scan with Swin-style window attention, gated by a content-aware fusion.

Architecture::

    Input (B, in_c, H, W)
      │
      ▼
    Stem ──────────────────────────────────────────────────────┐
      │                                                         │
      ▼                                                         │
    EncStage[0] ─── skip ──────────────────────────────────┐   │
      │  RMTBlock(s) → GatedFusion → Downsample             │   │
      ▼                                                     │   │
    EncStage[1] ─── skip ─────────────────────────────┐    │   │
      │                                                 │    │   │
      ▼                                                 │    │   │
    EncStage[2] ─── skip ────────────────────────┐    │    │   │
      │                                            │    │    │   │
      ▼                                            │    │    │   │
    Bottleneck  (RMTBlock(s), no downsample)       │    │    │   │
      │                                            │    │    │   │
      ▼                                            │    │    │   │
    DecStage[2] ◄── skip ─────────────────────────┘    │    │   │
      │  Upsample → Concat → ConvBlock                  │    │   │
      ▼                                                 │    │   │
    DecStage[1] ◄── skip ──────────────────────────────┘    │   │
      │                                                      │   │
      ▼                                                      │   │
    DecStage[0] ◄── skip ───────────────────────────────────┘   │
      │                                                          │
      ▼                                                          │
    SegHead ◄── stem skip ──────────────────────────────────────┘
      │
      ▼
    Output (B, num_classes, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rel_mamba import RelMambaBlock
from .local_transformer import LocalTransformerBlock
from .gated_fusion import GatedFusion, GatedFusionV2


# ======================================================================
#  Stem Block
# ======================================================================

class StemBlock(nn.Module):
    """
    Lightweight stem that projects raw range-view channels into the embedding
    space while (optionally) reducing spatial resolution.

    Two consecutive 3×3 convolutions with BN + SiLU provide initial local
    feature extraction without heavy computation.

    Parameters
    ----------
    in_channels : int
        Raw input channels  (e.g. 5 for x, y, z, range, intensity).
    embed_dim : int
        Output channel dimension.
    stem_stride : int, default=1
        Stride of the *first* convolution.  Set to 2 to halve resolution
        early for high-resolution inputs.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        stem_stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3,
                      stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)

        Returns
        -------
        (B, embed_dim, H_out, W_out)
        """
        return self.conv(x)


# ======================================================================
#  RMT  Block  (Rel Mamba + Local Transformer → Gated Fusion)
# ======================================================================

class RMTBlock(nn.Module):
    """
    One Rel-Mamba-Transformer fusion block.

    Runs a global-scope ``RelMambaBlock`` and a local-scope
    ``LocalTransformerBlock`` **in parallel**, then fuses their outputs with
    a content-aware ``GatedFusion`` (or ``GatedFusionV2``) module.

    Parameters
    ----------
    dim : int
        Channel dimension  (input == output).
    mamba_d_state : int, default=16
        SSM state size for RelMambaBlock.
    mamba_expand : int, default=2
        Channel expansion ratio inside RelMambaBlock.
    num_heads : int, default=8
        Attention heads for the local transformer.
    window_size : int, default=7
        Local window size.
    shift_size : int, default=0
        Shift amount for SW-MSA  (0 = W-MSA,  3 = SW-MSA for ws=7).
    mlp_ratio : float, default=4.0
        MLP expansion ratio in the transformer block.
    use_mamba_ssm : bool, default=False
        Passed to RelMambaBlock.
    gate_v2 : bool, default=False
        If True, use ``GatedFusionV2`` (bottleneck gate) instead of the
        plain ``GatedFusion``.
    attn_drop : float, default=0.0
        Attention dropout.
    proj_drop : float, default=0.0
        Projection dropout.
    drop_path : float, default=0.0
        Stochastic depth rate  (applied to both branches).
    """

    def __init__(
        self,
        dim: int,
        mamba_d_state: int = 16,
        mamba_expand: int = 2,
        num_heads: int = 8,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        use_mamba_ssm: bool = False,
        gate_v2: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        # ---- Global branch: multi-directional SSM -------------------------
        self.mamba = RelMambaBlock(
            dim=dim,
            d_state=mamba_d_state,
            expand=mamba_expand,
            use_mamba_ssm=use_mamba_ssm,
        )
        if drop_path > 0.0:
            self.mamba.drop_path = nn.Dropout(drop_path)

        # ---- Local branch: window-based attention -------------------------
        self.transformer = LocalTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            shift_size=shift_size,
            attn_drop=attn_drop,
            drop=proj_drop,
            drop_path=drop_path,
        )

        # ---- Gated fusion --------------------------------------------------
        gate_cls = GatedFusionV2 if gate_v2 else GatedFusion
        self.fusion = gate_cls(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        (B, C, H, W)  — fused feature map.
        """
        feat_global = self.mamba(x)
        feat_local = self.transformer(x)
        return self.fusion(feat_global, feat_local)


# ======================================================================
#  Encoder Stage
# ======================================================================

class EncoderStage(nn.Module):
    """
    One encoder stage: ``num_blocks`` RMT blocks → spatial downsampling.

    The feature map **before** downsampling is saved as the skip connection
    for the corresponding decoder stage.

    Parameters
    ----------
    dim : int
        Input channel dimension.
    out_dim : int
        Output channel dimension after downsampling.
    num_blocks : int, default=1
        Number of consecutive ``RMTBlock`` instances in this stage.
    downsample : bool, default=True
        If False, omit the downsampling layer  (used for the bottleneck).
    **rmt_kwargs
        Forwarded to every ``RMTBlock`` in this stage.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_blocks: int = 1,
        downsample: bool = True,
        **rmt_kwargs,
    ) -> None:
        super().__init__()

        # ---- RMT blocks ----------------------------------------------------
        blocks = []
        for i in range(num_blocks):
            block = RMTBlock(dim=dim, **rmt_kwargs)
            # Alternate W-MSA / SW-MSA across blocks within the stage
            if i % 2 == 1 and 'window_size' in rmt_kwargs:
                ws = rmt_kwargs['window_size']
                block.transformer.shift_size = ws // 2
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

        # ---- Downsampling --------------------------------------------------
        self.downsample = downsample
        if downsample:
            self.down_conv = nn.Sequential(
                nn.Conv2d(dim, out_dim, kernel_size=3, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(inplace=True),
            )
        else:
            # Bottleneck: project channels without spatial change
            self.proj = (
                nn.Conv2d(dim, out_dim, kernel_size=1, bias=False)
                if dim != out_dim else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, dim, H, W)

        Returns
        -------
        out  : (B, out_dim, H', W')  — downsampled (or projected) feature map.
        skip : (B, dim, H, W)        — pre-downsample feature for skip connection.
        """
        x = self.blocks(x)                                # (B, dim, H, W)
        skip = x

        if self.downsample:
            x = self.down_conv(x)                         # (B, out_dim, H/2, W/2)
        else:
            x = self.proj(x)                              # (B, out_dim, H, W)
        return x, skip


# ======================================================================
#  Decoder Stage
# ======================================================================

class DecoderStage(nn.Module):
    """
    One decoder stage: upsample → concat skip → double-conv fusion.

    Upsamples the incoming feature map by 2×, projects it to match the skip
    connection's channel count, concatenates, and refines with two 3×3 convs.

    Parameters
    ----------
    in_dim : int
        Channels from the previous (deeper) decoder stage or bottleneck.
    skip_dim : int
        Channels of the skip connection from the corresponding encoder stage.
    out_dim : int
        Output channel dimension.
    """

    def __init__(self, in_dim: int, skip_dim: int, out_dim: int) -> None:
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False,
        )
        self.up_conv = nn.Conv2d(in_dim, skip_dim, kernel_size=1, bias=False)
        self.bn_up = nn.BatchNorm2d(skip_dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(skip_dim * 2, out_dim, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, in_dim, H/2, W/2)
        skip : (B, skip_dim, H, W)

        Returns
        -------
        (B, out_dim, H, W)
        """
        x = self.upsample(x)                               # (B, in_dim, H, W)
        x = self.up_conv(x)                                # (B, skip_dim, H, W)
        x = self.bn_up(x)
        x = F.silu(x, inplace=True)

        x = torch.cat([x, skip], dim=1)                    # (B, 2·skip_dim, H, W)
        x = self.fuse(x)                                   # (B, out_dim, H, W)
        return x


# ======================================================================
#  Segmentation Head
# ======================================================================

class SegHead(nn.Module):
    """
    Lightweight segmentation head: 3×3 refine → 1×1 classifier.

    Optionally fuses the stem output as a low-level detail skip before the
    final prediction.

    Parameters
    ----------
    in_dim : int
        Feature dimension from the last decoder stage.
    stem_dim : int
        Stem output dimension  (for the optional stem skip).
    num_classes : int
        Number of semantic classes.
    stem_skip : bool, default=True
        If True, concatenate the stem feature map before the final conv.
    dropout : float, default=0.1
        Dropout rate before the classifier.
    """

    def __init__(
        self,
        in_dim: int,
        stem_dim: int,
        num_classes: int,
        stem_skip: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem_skip = stem_skip

        fuse_dim = (in_dim + stem_dim) if stem_skip else in_dim

        self.refine = nn.Sequential(
            nn.Conv2d(fuse_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.SiLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor,
                stem_feat: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_dim, H, W)
        stem_feat : (B, stem_dim, H, W) | None

        Returns
        -------
        (B, num_classes, H, W)  — raw logits.
        """
        if self.stem_skip and stem_feat is not None:
            # Align spatial resolution if stem used stride > 1
            if stem_feat.shape[2:] != x.shape[2:]:
                stem_feat = F.interpolate(
                    stem_feat, size=x.shape[2:],
                    mode='bilinear', align_corners=False,
                )
            x = torch.cat([x, stem_feat], dim=1)

        x = self.refine(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# ======================================================================
#  RMTSeg  — Full Network
# ======================================================================

class RMTSeg(nn.Module):
    """
    RMTSeg: Range-view Mamba-Transformer Segmentation Network.

    Parameters
    ----------
    in_channels : int, default=5
        Raw input channels  (e.g. x, y, z, range, intensity).
    num_classes : int, default=20
        Number of semantic categories.
    embed_dim : int, default=64
        Base channel dimension  (doubled at each encoder stage).
    num_stages : int, default=3
        Number of encoder-decoder stages  (produces 2^num_stages downsampling).
    num_blocks : int or list[int], default=1
        Number of ``RMTBlock`` per encoder stage.  Pass an int to use the
        same count for every stage, or a list of length ``num_stages`` for
        per-stage control.
    stem_stride : int, default=1
        Stride of the first stem convolution.
    stem_skip : bool, default=True
        Fuse stem features into the segmentation head.
    window_size : int, default=7
        Window size for the local transformer.
    num_heads : int, default=8
        Attention heads.
    mamba_d_state : int, default=16
        SSM state dimension.
    mamba_expand : int, default=2
        Channel expansion inside RelMambaBlock.
    mlp_ratio : float, default=4.0
        MLP hidden ratio in the transformer.
    use_mamba_ssm : bool, default=False
        Use the real ``mamba_ssm`` package instead of the Conv1d mock.
    gate_v2 : bool, default=False
        Use ``GatedFusionV2`` (bottleneck gate) in RMT blocks.
    attn_drop : float, default=0.0
        Attention dropout.
    proj_drop : float, default=0.0
        Projection dropout.
    drop_path_rate : float, default=0.0
        Max stochastic depth rate  (linearly scaled across encoder blocks).
    head_dropout : float, default=0.1
        Dropout rate in the segmentation head.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 20,
        embed_dim: int = 64,
        num_stages: int = 3,
        num_blocks: int | list[int] = 1,
        stem_stride: int = 1,
        stem_skip: bool = True,
        window_size: int = 7,
        num_heads: int = 8,
        mamba_d_state: int = 16,
        mamba_expand: int = 2,
        mlp_ratio: float = 4.0,
        use_mamba_ssm: bool = False,
        gate_v2: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ---- Normalise stage-block counts ----------------------------------
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks] * (num_stages + 1)  # +1 for bottleneck
        elif len(num_blocks) == num_stages:
            num_blocks = list(num_blocks) + [num_blocks[-1]]
        # Now ``num_blocks`` has length num_stages + 1

        # ---- Stem -----------------------------------------------------------
        self.stem = StemBlock(in_channels, embed_dim, stem_stride)

        # ---- Compute per-stage channel dimensions ---------------------------
        # enc_dims[i]: input channels for encoder stage i
        # dec_dims[i]: output channels for decoder stage i  (reversed)
        enc_dims = [embed_dim * (2 ** i) for i in range(num_stages)]
        bottleneck_dim = embed_dim * (2 ** num_stages)

        # ---- Stochastic depth schedule (linearly increasing) ----------------
        total_blocks = sum(num_blocks)
        dp_rates = [
            drop_path_rate * bid / max(total_blocks - 1, 1)
            for bid in range(total_blocks)
        ]

        # ---- Shared RMT kwargs ----------------------------------------------
        rmt_common = dict(
            mamba_d_state=mamba_d_state,
            mamba_expand=mamba_expand,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            use_mamba_ssm=use_mamba_ssm,
            gate_v2=gate_v2,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # ---- Encoder stages -------------------------------------------------
        block_idx = 0
        self.encoder_stages = nn.ModuleList()
        for i, in_dim in enumerate(enc_dims):
            out_dim = in_dim * 2
            n_blk = num_blocks[i]
            # Collect drop_path rates for the blocks in this stage
            stage_dp = dp_rates[block_idx:block_idx + n_blk]
            block_idx += n_blk
            self.encoder_stages.append(
                EncoderStage(
                    dim=in_dim,
                    out_dim=out_dim,
                    num_blocks=n_blk,
                    downsample=True,
                    drop_path=stage_dp[0] if stage_dp else 0.0,
                    **rmt_common,
                )
            )

            # Patch per-block drop_path — since we can't easily thread it
            # through the EncoderStage constructor for each block, we apply
            # the schedule by walking the built blocks.
            for j, blk in enumerate(self.encoder_stages[-1].blocks):
                dp = stage_dp[min(j, len(stage_dp) - 1)]
                blk.mamba.drop_path = nn.Dropout(dp) if dp > 0 else nn.Identity()
                blk.transformer.drop_path1 = nn.Dropout(dp) if dp > 0 else nn.Identity()
                blk.transformer.drop_path2 = nn.Dropout(dp) if dp > 0 else nn.Identity()

        # ---- Bottleneck -----------------------------------------------------
        n_blk_bn = num_blocks[-1]
        stage_dp = dp_rates[block_idx:block_idx + n_blk_bn]
        # Bottleneck may project channels if needed (but dims should match)
        self.bottleneck = EncoderStage(
            dim=bottleneck_dim,
            out_dim=bottleneck_dim,
            num_blocks=n_blk_bn,
            downsample=False,
            drop_path=stage_dp[0] if stage_dp else 0.0,
            **rmt_common,
        )
        for j, blk in enumerate(self.bottleneck.blocks):
            dp = stage_dp[min(j, len(stage_dp) - 1)]
            blk.mamba.drop_path = nn.Dropout(dp) if dp > 0 else nn.Identity()
            blk.transformer.drop_path1 = nn.Dropout(dp) if dp > 0 else nn.Identity()
            blk.transformer.drop_path2 = nn.Dropout(dp) if dp > 0 else nn.Identity()

        # ---- Decoder stages -------------------------------------------------
        # Decoder channels run in reverse: bottleneck → ... → embed_dim
        self.decoder_stages = nn.ModuleList()
        rev_enc_dims = list(reversed(enc_dims))  # [C₂, C₁, C₀] (largest first)
        dec_in_dim = bottleneck_dim
        for skip_dim in rev_enc_dims:
            self.decoder_stages.append(
                DecoderStage(in_dim=dec_in_dim, skip_dim=skip_dim,
                             out_dim=skip_dim)
            )
            dec_in_dim = skip_dim

        # ---- Segmentation head ----------------------------------------------
        self.head = SegHead(
            in_dim=embed_dim,
            stem_dim=embed_dim,
            num_classes=num_classes,
            stem_skip=stem_skip,
            dropout=head_dropout,
        )

        # ---- Weight initialisation ------------------------------------------
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Kaiming-normal init for Conv2d, constant init for BN."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)
            Range-view projection of a LiDAR point cloud.

        Returns
        -------
        (B, num_classes, H, W)
            Per-pixel logits for semantic segmentation.
        """
        # ---- Stem -----------------------------------------------------------
        stem_feat = self.stem(x)                           # (B, embed_dim, H', W')

        # ---- Encoder --------------------------------------------------------
        feats = stem_feat
        skips: list[torch.Tensor] = []
        for stage in self.encoder_stages:
            feats, skip = stage(feats)                     # out, skip
            skips.append(skip)

        # ---- Bottleneck -----------------------------------------------------
        feats, _ = self.bottleneck(feats)                  # (B, bn_dim, Hₙ, Wₙ)

        # ---- Decoder --------------------------------------------------------
        for i, stage in enumerate(self.decoder_stages):
            feats = stage(feats, skips[-(i + 1)])          # skip in reverse order

        # ---- Head -----------------------------------------------------------
        out = self.head(feats, stem_feat)                  # (B, num_classes, H', W')

        return out


# ======================================================================
#  Convenience aliases
# ======================================================================

def rmt_seg_tiny(in_channels: int = 5, num_classes: int = 20, **kwargs) -> RMTSeg:
    """Tiny variant: embed_dim=32, 3 stages, 1 block each."""
    return RMTSeg(in_channels=in_channels, num_classes=num_classes,
                  embed_dim=32, num_stages=3, num_blocks=1, **kwargs)


def rmt_seg_small(in_channels: int = 5, num_classes: int = 20, **kwargs) -> RMTSeg:
    """Small variant: embed_dim=64, 3 stages, 1 block each."""
    return RMTSeg(in_channels=in_channels, num_classes=num_classes,
                  embed_dim=64, num_stages=3, num_blocks=1, **kwargs)


def rmt_seg_base(in_channels: int = 5, num_classes: int = 20, **kwargs) -> RMTSeg:
    """Base variant: embed_dim=64, 4 stages, [1,1,2,1] blocks."""
    return RMTSeg(in_channels=in_channels, num_classes=num_classes,
                  embed_dim=64, num_stages=4, num_blocks=[1, 1, 2, 1], **kwargs)
