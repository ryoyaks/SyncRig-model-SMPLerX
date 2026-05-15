"""ViT backbone with task tokens — port of upstream `mmpose` ViT.

State-dict layer names match upstream exactly so checkpoints load with
``strict=True`` after the ``module.encoder.`` prefix is stripped:

    patch_embed.proj.{weight,bias}
    pos_embed                              shape (1, num_patches+1, D)
    task_tokens                            shape (1, task_tokens_num, D)
    blocks.{i}.norm{1,2}.{weight,bias}
    blocks.{i}.attn.qkv.{weight,bias}
    blocks.{i}.attn.proj.{weight,bias}
    blocks.{i}.mlp.fc{1,2}.{weight,bias}
    last_norm.{weight,bias}

The upstream class also carried freeze_attn / freeze_ffn / DropPath
machinery that's irrelevant to inference; we drop those branches.
DropPath is replaced with `nn.Identity()` because we run in `eval()`
mode for inference (its `drop_path_rate > 0` is only a training-time
thing).

For the position embedding the upstream forward subtracts the cls
token's slot at index 0 then adds the patch embedding to positions
[1:]. We replicate that arithmetic exactly so the trained weights
keep producing the same activations:

    x = patch_embed(x) + pos_embed[:, 1:] + pos_embed[:, :1]
    x = cat([task_tokens, x], dim=1)
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(v):
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v, v)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float,
                 qkv_bias: bool, norm_layer):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = nn.Identity()  # inference-time no-op
        self.norm2 = norm_layer(dim)
        self.mlp = _Mlp(dim, hidden_dim=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    """Conv2d patch embedder. ratio=1 (the only setting we support)."""

    def __init__(self, img_size: tuple[int, int], patch_size: int,
                 in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        h, w = img_size
        # Upstream uses padding=4 for the ratio=1 case (12 patches at
        # patch_size=16 over 192 input width: 4 + 192/16 = 16 → that's
        # 12 patches, not 16. Investigated upstream: with padding=4 the
        # first conv produces a (Hp, Wp) = ((H+8)/16, (W+8)/16) grid =
        # (16, 12) for the (256, 192) body input. The pos_embed is sized
        # to 192 + 1 = 193 entries to match.
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
            padding=4 + 2 * (1 // 2 - 1),  # ratio=1 → padding=4
        )

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class ViT(nn.Module):
    """ViT backbone with task tokens for SMPLer-X HPS heads.

    Returned by `forward`:
      img_feat:    (B, embed_dim, Hp, Wp) image-token map (post-norm)
      task_tokens: (B, task_tokens_num, embed_dim) head tokens
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (256, 192),
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        task_tokens_num: int = 31,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.task_tokens_num = task_tokens_num

        self.patch_embed = _PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # num_patches = 192 for (256, 192) ViT-S, plus the +1 cls slot
        # the upstream pos_embed was sized to inherit pretrained weights.
        # We compute it lazily after a dummy forward.
        h, w = img_size
        ratio_pad = 4
        Hp = (h + 2 * ratio_pad) // patch_size
        Wp = (w + 2 * ratio_pad) // patch_size
        num_patches = Hp * Wp

        self.task_tokens = nn.Parameter(torch.zeros(1, task_tokens_num, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
            _Block(embed_dim, num_heads, mlp_ratio, qkv_bias, norm_layer)
            for _ in range(depth)
        ])
        self.last_norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)
        # Upstream pattern: x = x + pos[:, 1:] + pos[:, :1] (the first
        # entry is the unused cls token; adding it as a global offset
        # matches the trained weights).
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        task_tokens = self.task_tokens.expand(B, -1, -1)
        x = torch.cat((task_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)

        task_tokens = x[:, : self.task_tokens_num]
        xp = x[:, self.task_tokens_num :]
        xp = xp.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return xp, task_tokens
