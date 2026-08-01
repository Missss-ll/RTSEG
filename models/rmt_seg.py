
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rel_mamba import RelMambaBlock
from .local_transformer import LocalTransformerBlock
from .gated_fusion import GatedFusion, GatedFusionV2
from utils.window_select import AdaptiveLocalTransformer

class StemBlock(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int,
                 stem_stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, stride=stem_stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class RMTBlock(nn.Module):

    def __init__(self, dim: int, mamba_d_state: int = 16, mamba_expand: int = 2,
                 num_heads: int = 8, window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4.0, use_mamba_ssm: bool = False,
                 gate_v2: bool = False, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, drop_path: float = 0.0,
                 adaptive_window: bool = False,
                 window_keep_ratio: float = 0.7) -> None:
        super().__init__()

        self.mamba = RelMambaBlock(dim=dim, d_state=mamba_d_state,
                                   expand=mamba_expand,
                                   use_mamba_ssm=use_mamba_ssm)
        if drop_path > 0.0:
            self.mamba.drop_path = nn.Dropout(drop_path)

        if adaptive_window:
            self.transformer = AdaptiveLocalTransformer(
                dim=dim, num_heads=num_heads, window_size=window_size,
                keep_ratio=window_keep_ratio, shift_size=shift_size,
                mlp_ratio=mlp_ratio, drop=proj_drop, attn_drop=attn_drop,
                drop_path=drop_path,
            )
        else:
            self.transformer = LocalTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                mlp_ratio=mlp_ratio, shift_size=shift_size, attn_drop=attn_drop,
                drop=proj_drop, drop_path=drop_path,
            )

        self.adaptive_window = adaptive_window
        gate_cls = GatedFusionV2 if gate_v2 else GatedFusion
        self.fusion = gate_cls(dim=dim)

    def forward(self, x: torch.Tensor,
                difficulty_map: torch.Tensor | None = None) -> torch.Tensor:
        f_global = self.mamba(x)
        if self.adaptive_window:
            f_local = self.transformer(x, difficulty_map)
        else:
            f_local = self.transformer(x)
        return self.fusion(f_global, f_local)

class EncoderStage(nn.Module):

    def __init__(self, dim: int, out_dim: int, num_blocks: int = 1,
                 downsample: bool = True, **rmt_kwargs) -> None:
        super().__init__()

        blocks = []
        for i in range(num_blocks):
            blk = RMTBlock(dim=dim, **rmt_kwargs)
            if i % 2 == 1 and 'window_size' in rmt_kwargs:
                blk.transformer.shift_size = rmt_kwargs['window_size'] // 2
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)

        self.downsample = downsample
        if downsample:
            self.down_conv = nn.Sequential(
                nn.Conv2d(dim, out_dim, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(inplace=True),
            )
        else:
            self.proj = (nn.Conv2d(dim, out_dim, 1, bias=False)
                         if dim != out_dim else nn.Identity())

    def forward(self, x: torch.Tensor,
                difficulty_map: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            x = blk(x, difficulty_map)
        skip = x
        if self.downsample:
            x = self.down_conv(x)
        else:
            x = self.proj(x)
        return x, skip

class DecoderStage(nn.Module):

    def __init__(self, in_dim: int, skip_dim: int, out_dim: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=False)
        self.up_conv = nn.Conv2d(in_dim, skip_dim, 1, bias=False)
        self.bn_up = nn.BatchNorm2d(skip_dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(skip_dim * 2, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.SiLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = F.silu(self.bn_up(self.up_conv(x)), inplace=True)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

class SegHead(nn.Module):

    def __init__(self, in_dim: int, stem_dim: int, num_classes: int,
                 stem_skip: bool = True, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem_skip = stem_skip
        fuse_dim = (in_dim + stem_dim) if stem_skip else in_dim

        self.refine = nn.Sequential(
            nn.Conv2d(fuse_dim, in_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim), nn.SiLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(in_dim, num_classes, 1)

    def forward(self, x: torch.Tensor,
                stem_feat: torch.Tensor | None = None) -> torch.Tensor:
        if self.stem_skip and stem_feat is not None:
            if stem_feat.shape[2:] != x.shape[2:]:
                stem_feat = F.interpolate(stem_feat, size=x.shape[2:],
                                          mode='bilinear', align_corners=False)
            x = torch.cat([x, stem_feat], dim=1)
        x = self.refine(x)
        x = self.dropout(x)
        return self.classifier(x)

class RMTSeg(nn.Module):

    def __init__(self, in_channels: int = 5, num_classes: int = 20,
                 embed_dim: int = 64, num_stages: int = 3,
                 num_blocks: int | list[int] = 1, stem_stride: int = 1,
                 stem_skip: bool = True, window_size: int = 7,
                 num_heads: int = 8, mamba_d_state: int = 16,
                 mamba_expand: int = 2, mlp_ratio: float = 4.0,
                 use_mamba_ssm: bool = False, gate_v2: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 drop_path_rate: float = 0.0,
                 head_dropout: float = 0.1,
                 adaptive_window: bool = False,
                 window_keep_ratio: float = 0.7) -> None:
        super().__init__()

        if isinstance(num_blocks, int):
            num_blocks = [num_blocks] * (num_stages + 1)
        elif len(num_blocks) == num_stages:
            num_blocks = list(num_blocks) + [num_blocks[-1]]

        self.stem = StemBlock(in_channels, embed_dim, stem_stride)

        enc_dims = [embed_dim * (2 ** i) for i in range(num_stages)]
        bottleneck_dim = embed_dim * (2 ** num_stages)

        total_blocks = sum(num_blocks)
        dp_rates = [drop_path_rate * i / max(total_blocks - 1, 1)
                    for i in range(total_blocks)]

        rmt_common = dict(mamba_d_state=mamba_d_state, mamba_expand=mamba_expand,
                          num_heads=num_heads, window_size=window_size,
                          mlp_ratio=mlp_ratio, use_mamba_ssm=use_mamba_ssm,
                          gate_v2=gate_v2, attn_drop=attn_drop,
                          proj_drop=proj_drop,
                          adaptive_window=adaptive_window,
                          window_keep_ratio=window_keep_ratio)

        block_idx = 0
        self.encoder_stages = nn.ModuleList()
        for i, in_dim in enumerate(enc_dims):
            out_dim = in_dim * 2
            n_blk = num_blocks[i]
            stage_dp = dp_rates[block_idx:block_idx + n_blk]
            block_idx += n_blk
            self.encoder_stages.append(
                EncoderStage(dim=in_dim, out_dim=out_dim, num_blocks=n_blk,
                             downsample=True,
                             drop_path=stage_dp[0] if stage_dp else 0.0,
                             **rmt_common))
            for j, blk in enumerate(self.encoder_stages[-1].blocks):
                dp = stage_dp[min(j, len(stage_dp) - 1)]
                blk.mamba.drop_path = nn.Dropout(dp) if dp > 0 else nn.Identity()
                blk.transformer.drop_path1 = nn.Dropout(dp) if dp > 0 else nn.Identity()
                blk.transformer.drop_path2 = nn.Dropout(dp) if dp > 0 else nn.Identity()

        n_blk_bn = num_blocks[-1]
        stage_dp = dp_rates[block_idx:block_idx + n_blk_bn]
        self.bottleneck = EncoderStage(
            dim=bottleneck_dim, out_dim=bottleneck_dim, num_blocks=n_blk_bn,
            downsample=False,
            drop_path=stage_dp[0] if stage_dp else 0.0, **rmt_common)
        for j, blk in enumerate(self.bottleneck.blocks):
            dp = stage_dp[min(j, len(stage_dp) - 1)]
            blk.mamba.drop_path = nn.Dropout(dp) if dp > 0 else nn.Identity()
            blk.transformer.drop_path1 = nn.Dropout(dp) if dp > 0 else nn.Identity()
            blk.transformer.drop_path2 = nn.Dropout(dp) if dp > 0 else nn.Identity()

        self.decoder_stages = nn.ModuleList()
        dec_in_dim = bottleneck_dim
        for skip_dim in reversed(enc_dims):
            self.decoder_stages.append(
                DecoderStage(in_dim=dec_in_dim, skip_dim=skip_dim,
                             out_dim=skip_dim))
            dec_in_dim = skip_dim

        self.head = SegHead(in_dim=embed_dim, stem_dim=embed_dim,
                            num_classes=num_classes, stem_skip=stem_skip,
                            dropout=head_dropout)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
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

    def forward(self, x: torch.Tensor,
                difficulty_map: torch.Tensor | None = None) -> torch.Tensor:
        stem_feat = self.stem(x)

        feats = stem_feat
        skips: list[torch.Tensor] = []
        for stage in self.encoder_stages:
            feats, skip = stage(feats, difficulty_map)
            skips.append(skip)

        feats, _ = self.bottleneck(feats, difficulty_map)

        for i, stage in enumerate(self.decoder_stages):
            feats = stage(feats, skips[-(i + 1)])

        return self.head(feats, stem_feat)

def rmt_seg_tiny(in_channels: int = 5, num_classes: int = 20, **kwargs) -> RMTSeg:
    return RMTSeg(in_channels=in_channels, num_classes=num_classes,
                  embed_dim=32, num_stages=3, num_blocks=1, **kwargs)

def rmt_seg_small(in_channels: int = 5, num_classes: int = 20, **kwargs) -> RMTSeg:
    return RMTSeg(in_channels=in_channels, num_classes=num_classes,
                  embed_dim=64, num_stages=3, num_blocks=1, **kwargs)

def rmt_seg_base(in_channels: int = 5, num_classes: int = 20, **kwargs) -> RMTSeg:
    return RMTSeg(in_channels=in_channels, num_classes=num_classes,
                  embed_dim=64, num_stages=4,
                  num_blocks=[1, 1, 2, 1], **kwargs)