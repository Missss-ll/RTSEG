
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.local_transformer import WindowAttention

def compute_difficulty_map(
    reliability: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    if reliability.dim() == 4:
        reliability = reliability.squeeze(1)
    if reliability.dim() == 2:
        reliability = reliability.unsqueeze(0).unsqueeze(0)
    elif reliability.dim() == 3:
        reliability = reliability.unsqueeze(1)

    R_pooled = F.adaptive_avg_pool2d(reliability, (target_h, target_w)).squeeze(1)
    return (1.0 - R_pooled).clamp(0.0, 1.0)

def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, ws, ws, C)

def _window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    nH, nW = H // ws, W // ws
    B = windows.shape[0] // (nH * nW)
    x = windows.view(B, nH, nW, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)

class AdaptiveLocalTransformer(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        keep_ratio: float = 0.7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.keep_ratio = keep_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim, num_heads=num_heads, window_size=window_size,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path1 = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )
        self.drop_path2 = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

        self.light_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

        self._cached_mask: torch.Tensor | None = None

    def _build_shifted_mask(
        self, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        ws = self.window_size
        sh = self.shift_size
        img_mask = torch.zeros((1, H, W, 1), device=device, dtype=dtype)
        h_slices = (slice(0, -ws), slice(-ws, -sh), slice(-sh, None))
        w_slices = (slice(0, -ws), slice(-ws, -sh), slice(-sh, None))
        cnt = 0
        for hs in h_slices:
            for wsl in w_slices:
                img_mask[:, hs, wsl, :] = cnt
                cnt += 1
        mask_windows = _window_partition(img_mask, ws)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(
        self,
        x: torch.Tensor,
        difficulty_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size
        sh = self.shift_size

        if difficulty_map is None or self.keep_ratio >= 1.0:
            return self._full_attention_forward(x)

        x = x.permute(0, 2, 3, 1).contiguous()
        Hp = ((H + ws - 1) // ws) * ws
        Wp = ((W + ws - 1) // ws) * ws
        pad_h, pad_w = Hp - H, Wp - W

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            diff = F.pad(difficulty_map, (0, pad_w, 0, pad_h))
        else:
            diff = difficulty_map

        if sh > 0:
            x = torch.roll(x, shifts=(-sh, -sh), dims=(1, 2))
            diff = torch.roll(diff, shifts=(-sh, -sh), dims=(1, 2))
            if self._cached_mask is None:
                self._cached_mask = self._build_shifted_mask(Hp, Wp, x.device, x.dtype)
            attn_mask = self._cached_mask
        else:
            attn_mask = None

        x_wins = _window_partition(x, ws)
        N_win = x_wins.shape[0]
        x_wins_flat = x_wins.view(N_win, ws * ws, C)

        nH, nW = Hp // ws, Wp // ws
        d = diff.view(B, nH, ws, nW, ws)
        d = d.permute(0, 1, 3, 2, 4).contiguous()
        mean_d = d.view(B, nH * nW, ws * ws).mean(dim=-1).view(-1)

        top_k = max(1, int(N_win * self.keep_ratio))
        _, sorted_idx = torch.sort(mean_d, descending=True)
        hard_idx = sorted_idx[:top_k]
        easy_idx = sorted_idx[top_k:]

        shortcut = x_wins_flat
        x_normed = self.norm1(x_wins_flat)
        out = shortcut.clone()

        if hard_idx.numel() > 0:
            hard_in = x_normed[hard_idx]
            hard_mask = attn_mask[hard_idx] if attn_mask is not None else None
            hard_out = self.attn(hard_in, hard_mask)
            out[hard_idx] = shortcut[hard_idx] + self.drop_path1(hard_out)

        if easy_idx.numel() > 0:
            easy_win = x_wins[easy_idx]
            easy_bchw = easy_win.permute(0, 3, 1, 2)
            easy_out = self.light_conv(easy_bchw)
            easy_flat = easy_out.permute(0, 2, 3, 1).reshape(
                -1, ws * ws, C
            )
            out[easy_idx] = easy_flat

        out = out + self.drop_path2(self.mlp(self.norm2(out)))

        out = out.view(N_win, ws, ws, C)
        x_out = _window_reverse(out, ws, Hp, Wp)
        if sh > 0:
            x_out = torch.roll(x_out, shifts=(sh, sh), dims=(1, 2))
        if pad_h > 0 or pad_w > 0:
            x_out = x_out[:, :H, :W, :]

        return x_out.permute(0, 3, 1, 2).contiguous()

    def _full_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size
        sh = self.shift_size

        x = x.permute(0, 2, 3, 1).contiguous()
        Hp = ((H + ws - 1) // ws) * ws
        Wp = ((W + ws - 1) // ws) * ws
        pad_h, pad_w = Hp - H, Wp - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        if sh > 0:
            x = torch.roll(x, shifts=(-sh, -sh), dims=(1, 2))
            if self._cached_mask is None:
                self._cached_mask = self._build_shifted_mask(Hp, Wp, x.device, x.dtype)
            mask = self._cached_mask
        else:
            mask = None

        blk_shortcut = x
        x = self.norm1(x)
        x_wins = _window_partition(x, ws)
        x_flat = x_wins.view(-1, ws * ws, C)
        x_flat = self.attn(x_flat, mask)
        x = x_flat.view(-1, ws, ws, C)
        x = _window_reverse(x, ws, Hp, Wp)
        if sh > 0:
            x = torch.roll(x, shifts=(sh, sh), dims=(1, 2))
        x = blk_shortcut + self.drop_path1(x)

        mlp_shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = mlp_shortcut + self.drop_path2(x)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]

        return x.permute(0, 3, 1, 2).contiguous()