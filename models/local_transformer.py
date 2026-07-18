"""
LocalTransformerBlock: Window-based Local Attention Module.

Implements window-based multi-head self-attention (W-MSA) and shifted-window
attention (SW-MSA) in the style of Swin Transformer, with learnable relative
position bias.

Reference:
    "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    (Liu et al., ICCV 2021)

Typical usage (alternating windows for cross-window communication)::

    block_0 = LocalTransformerBlock(dim, window_size=7, shift_size=0)   # W-MSA
    block_1 = LocalTransformerBlock(dim, window_size=7, shift_size=3)   # SW-MSA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
#  Window-based Multi-Head Self-Attention
# ======================================================================

class WindowAttention(nn.Module):
    """
    Multi-head self-attention confined to local windows, with learnable
    relative-position bias (Swin-style).

    Parameters
    ----------
    dim : int
        Total channel dimension of the input features.
    num_heads : int
        Number of attention heads  (``dim`` must be divisible by this).
    window_size : int
        Spatial size of the square window  (default 7).
    qkv_bias : bool
        If True, include bias terms in the QKV linear projection.
    attn_drop : float
        Dropout rate applied to attention weights.
    proj_drop : float
        Dropout rate applied to the output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.N = window_size * window_size  # number of tokens per window

        # ---- Learnable relative-position bias --------------------------------
        # For a window of size ws×ws, relative co-ordinates along each axis
        # range from -(ws-1) to (ws-1) ⇒ (2·ws-1) distinct values per axis.
        n_rel = (2 * window_size - 1) ** 2
        self.rel_pos_bias = nn.Parameter(torch.zeros(n_rel, num_heads))
        self._register_rel_pos_index(window_size)

        # ---- Linear projections ---------------------------------------------
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _register_rel_pos_index(self, ws: int) -> None:
        """
        Pre-compute the relative-position lookup indices for a ws×ws window.

        The index table maps each query-key pair (i, j) within the flattened
        window to an entry in ``rel_pos_bias``.
        """
        # Co-ordinate grids
        coords = torch.stack(
            torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij')
        )                                                        # (2, ws, ws)
        coords_flat = coords.flatten(1)                          # (2, ws²)

        # Pair-wise relative co-ordinates  (2, ws², ws²)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0)                               # (ws², ws², 2)

        # Shift to non-negative range  [0, 2·ws-2]
        rel += ws - 1
        # Flatten the 2-D relative index to 1-D
        rel[:, :, 0] *= (2 * ws - 1)
        rel_idx = rel.sum(dim=-1)                                # (ws², ws²)

        self.register_buffer('rel_pos_index', rel_idx, persistent=False)

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (nW·B, N, C)  where  N = window_size²,  nW = total windows
        attn_mask : (nW, N, N) | None
            Optional mask for shifted-window attention; entries that should
            *not* attend to each other are set to a large negative value.

        Returns
        -------
        (nW·B, N, C)  — same shape as input.
        """
        B_, N, C = x.shape
        # ---- QKV projection & multi-head reshape ----------------------------
        qkv = self.qkv(x) \
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads) \
            .permute(2, 0, 3, 1, 4)                             # (3, B_, nH, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]                        # (B_, nH, N, d)

        # ---- Scaled dot-product attention -----------------------------------
        attn = (q * self.scale) @ k.transpose(-2, -1)            # (B_, nH, N, N)

        # Relative position bias
        bias = self.rel_pos_bias[self.rel_pos_index.view(-1)] \
            .view(N, N, self.num_heads) \
            .permute(2, 0, 1).unsqueeze(0)                       # (1, nH, N, N)
        attn = attn + bias

        # Shifted-window mask  (if provided)
        if attn_mask is not None:
            nW = attn_mask.shape[0]                              # total windows
            # attn: (nW·B, nH, N, N)  →  (B, nW, nH, N, N)
            attn = attn.view(-1, nW, self.num_heads, N, N)
            # mask:  (nW, N, N) → (1, nW, 1, N, N)  → broadcasts
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # ---- Weighted sum + output projection -------------------------------
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ======================================================================
#  Local Transformer Block  (W-MSA / SW-MSA)
# ======================================================================

class LocalTransformerBlock(nn.Module):
    """
    Window-based local transformer block.

    Supports two operating modes selected via ``shift_size``:

    * **W-MSA**  (``shift_size=0``)  — regular non-overlapping windows.
    * **SW-MSA** (``shift_size>0``) — cyclic-shifted windows that enable
      cross-window information flow.

    Each sub-block = LayerNorm → WindowAttention/MLP → residual.

    Parameters
    ----------
    dim : int
        Input / output channel dimension.
    num_heads : int, default=8
        Number of attention heads.
    window_size : int, default=7
        Spatial size of each square window.
    mlp_ratio : float, default=4.0
        Expansion ratio for the MLP hidden layer  (hidden = dim * mlp_ratio).
    shift_size : int, default=0
        Cyclic shift amount; set to ``window_size // 2`` for SW-MSA.
    qkv_bias : bool, default=True
        Bias flag for the QKV projection.
    drop : float, default=0.0
        Dropout rate in the MLP and attention projection.
    attn_drop : float, default=0.0
        Dropout rate on attention weights.
    drop_path : float, default=0.0
        Stochastic depth rate  (set > 0 when using DropPath).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        shift_size: int = 0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # ---- Attention sub-block --------------------------------------------
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = (
            nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        # ---- MLP sub-block --------------------------------------------------
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
        self.drop_path2 = (
            nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        # Cached shifted-window attention mask  (built on first forward)
        self._cached_mask: torch.Tensor | None = None

    # ------------------------------------------------------------------
    #  Window partition / reverse  (static helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
        """
        Split a feature map into non-overlapping ws×ws windows.

        Parameters
        ----------
        x : (B, H, W, C)  — channel-last layout.
        ws : window size.

        Returns
        -------
        (B·nH·nW, ws, ws, C)  where  nH = H/ws,  nW = W/ws.
        """
        B, H, W, C = x.shape
        # (B, H, W, C) → (B, nH, ws, nW, ws, C)
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        # → (B, nH, nW, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # → (B·nH·nW, ws, ws, C)
        return x.view(-1, ws, ws, C)

    @staticmethod
    def _window_reverse(windows: torch.Tensor, ws: int,
                        H: int, W: int) -> torch.Tensor:
        """
        Merge windows back into a feature map  (inverse of ``_window_partition``).

        Parameters
        ----------
        windows : (B·nH·nW, ws, ws, C)
        ws : window size.
        H, W : original spatial resolution.

        Returns
        -------
        (B, H, W, C)
        """
        nH, nW = H // ws, W // ws
        B = windows.shape[0] // (nH * nW)
        # (·, ws, ws, C) → (B, nH, nW, ws, ws, C)
        x = windows.view(B, nH, nW, ws, ws, -1)
        # → (B, nH, ws, nW, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # → (B, H, W, C)
        return x.view(B, H, W, -1)

    # ------------------------------------------------------------------
    #  Shifted-window attention mask
    # ------------------------------------------------------------------

    def _build_shifted_mask(self, H: int, W: int,
                            device: torch.device,
                            dtype: torch.dtype) -> torch.Tensor:
        """
        Build the attention mask for cyclic-shifted window attention.

        Pixels belonging to different *original* windows are assigned distinct
        region IDs, and the mask blocks cross-region attention by adding a
        large negative value.

        Parameters
        ----------
        H, W : padded spatial resolution (multiples of window_size).
        device, dtype : desired tensor properties.

        Returns
        -------
        (nW, N, N)  mask, where  nW = total windows,  N = window_size².
        """
        ws = self.window_size
        sh = self.shift_size

        # Create an image whose pixels carry a region label (0 … 8)
        img_mask = torch.zeros((1, H, W, 1), device=device, dtype=dtype)
        # Three horizontal slices  (top, middle, bottom)
        h_slices = (slice(0, -ws), slice(-ws, -sh), slice(-sh, None))
        # Three vertical slices    (left, middle, right)
        w_slices = (slice(0, -ws), slice(-ws, -sh), slice(-sh, None))
        for cnt, (hs, ws_slice) in enumerate(
            (hs, ws_slice)
            for hs in h_slices
            for ws_slice in w_slices
        ):
            img_mask[:, hs, ws_slice, :] = cnt

        # Partition the label map into windows
        mask_windows = self._window_partition(img_mask, ws)      # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, ws * ws)            # (nW, N)

        # Pair-wise mask: equal labels → attend (0), different → mask (-100)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)  — channel-first feature map.

        Returns
        -------
        (B, C, H, W)  — same spatial resolution.
        """
        B, C, H, W = x.shape
        ws = self.window_size
        sh = self.shift_size

        # ---- Convert to channel-last  (Swin works in NHWC) ------------------
        x = x.permute(0, 2, 3, 1).contiguous()                  # (B, H, W, C)

        # ---- Pad spatial dims to multiples of window_size -------------------
        Hp = ((H + ws - 1) // ws) * ws
        Wp = ((W + ws - 1) // ws) * ws
        pad_h, pad_w = Hp - H, Wp - W
        if pad_h > 0 or pad_w > 0:
            # F.pad on 4-D: (left, right, top, bottom, front, back)
            #  →  (0, 0,  0, pad_w,  0, pad_h)  pads C, W, H
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        # ---- Cyclic shift  (SW-MSA) -----------------------------------------
        if sh > 0:
            x = torch.roll(x, shifts=(-sh, -sh), dims=(1, 2))
            if self._cached_mask is None:
                self._cached_mask = self._build_shifted_mask(
                    Hp, Wp, x.device, x.dtype
                )
            mask = self._cached_mask
        else:
            mask = None

        # ---- Attention sub-block --------------------------------------------
        blk_shortcut = x
        x = self.norm1(x)                                        # (B, Hp, Wp, C)
        x = self._window_partition(x, ws)                        # (nW·B, ws, ws, C)
        x = x.view(-1, ws * ws, C)                               # (nW·B, N, C)
        x = self.attn(x, mask)                                   # (nW·B, N, C)
        x = x.view(-1, ws, ws, C)                                # (nW·B, ws, ws, C)
        x = self._window_reverse(x, ws, Hp, Wp)                  # (B, Hp, Wp, C)
        if sh > 0:
            # Reverse the cyclic shift
            x = torch.roll(x, shifts=(sh, sh), dims=(1, 2))
        x = blk_shortcut + self.drop_path1(x)

        # ---- MLP sub-block --------------------------------------------------
        mlp_shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)                                          # (B, Hp, Wp, C)
        x = mlp_shortcut + self.drop_path2(x)

        # ---- Un-pad ---------------------------------------------------------
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]

        # ---- Convert back to channel-first ----------------------------------
        x = x.permute(0, 3, 1, 2).contiguous()                   # (B, C, H, W)

        return x
