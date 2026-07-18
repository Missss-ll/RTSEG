"""
Window-based local attention with shifted-window and relative position bias.
Follows the Swin Transformer local-attention pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """Multi-head self-attention within a local window, with learnable relative
    position bias table."""

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 7,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.N = window_size * window_size

        # Learnable relative-position bias: (2*ws-1)^2 entries per head
        n_rel = (2 * window_size - 1) ** 2
        self.rel_pos_bias = nn.Parameter(torch.zeros(n_rel, num_heads))
        self._register_rel_pos_index(window_size)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _register_rel_pos_index(self, ws: int) -> None:
        coords = torch.stack(
            torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij')
        )
        coords_flat = coords.flatten(1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0)
        rel += ws - 1
        rel[:, :, 0] *= (2 * ws - 1)
        rel_idx = rel.sum(dim=-1)
        self.register_buffer('rel_pos_index', rel_idx, persistent=False)

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.rel_pos_bias[self.rel_pos_index.view(-1)]
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias

        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalTransformerBlock(nn.Module):
    """Window-based transformer block with W-MSA / SW-MSA modes.

    shift_size = 0        → regular non-overlapping windows (W-MSA)
    shift_size = ws // 2  → cyclic-shifted windows (SW-MSA)
    """

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 7,
                 mlp_ratio: float = 4.0, shift_size: int = 0,
                 qkv_bias: bool = True, drop: float = 0.0,
                 attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, num_heads=num_heads,
                                    window_size=window_size, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )
        self.drop_path2 = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

        self._cached_mask: torch.Tensor | None = None

    # ------------------------------------------------------------------
    #  Window partition helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(-1, ws, ws, C)

    @staticmethod
    def _window_reverse(windows: torch.Tensor, ws: int,
                        H: int, W: int) -> torch.Tensor:
        nH, nW = H // ws, W // ws
        B = windows.shape[0] // (nH * nW)
        x = windows.view(B, nH, nW, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, H, W, -1)

    def _build_shifted_mask(self, H: int, W: int, device: torch.device,
                            dtype: torch.dtype) -> torch.Tensor:
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

        mask_windows = self._window_partition(img_mask, ws)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size
        sh = self.shift_size

        x = x.permute(0, 2, 3, 1).contiguous()  # BCHW → BHWC

        # Pad to multiples of window_size
        Hp = ((H + ws - 1) // ws) * ws
        Wp = ((W + ws - 1) // ws) * ws
        pad_h, pad_w = Hp - H, Wp - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        # Cyclic shift for SW-MSA
        if sh > 0:
            x = torch.roll(x, shifts=(-sh, -sh), dims=(1, 2))
            if self._cached_mask is None:
                self._cached_mask = self._build_shifted_mask(Hp, Wp, x.device, x.dtype)
            mask = self._cached_mask
        else:
            mask = None

        # W-MSA / SW-MSA
        blk_shortcut = x
        x = self.norm1(x)
        x = self._window_partition(x, ws)
        x = x.view(-1, ws * ws, C)
        x = self.attn(x, mask)
        x = x.view(-1, ws, ws, C)
        x = self._window_reverse(x, ws, Hp, Wp)
        if sh > 0:
            x = torch.roll(x, shifts=(sh, sh), dims=(1, 2))
        x = blk_shortcut + self.drop_path1(x)

        # MLP
        mlp_shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = mlp_shortcut + self.drop_path2(x)

        # Un-pad
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]

        x = x.permute(0, 3, 1, 2).contiguous()  # BHWC → BCHW
        return x
