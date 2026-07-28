"""MHA、GQA 与教学版 MLA 的形状和 KV-cache 对比。

运行：
    conda activate tf
    python code/attention_variants_demo.py

ToyMLA 是“先压缩、后恢复 K/V”的教学版，不是某个生产模型的完整 MLA 实现。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q/k/v: [B, H, T, D]；返回 [B, H, T, D]。"""
    _, _, time, dim = q.shape
    scores = q @ k.transpose(-2, -1) / math.sqrt(dim)
    mask = torch.tril(torch.ones(time, time, device=q.device, dtype=torch.bool))
    return F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v


class MHA(nn.Module):
    def __init__(self, d_model: int = 32, n_heads: int = 4) -> None:
        super().__init__()
        self.h, self.d = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        split = lambda z: z.view(batch, time, self.h, self.d).transpose(1, 2)
        y = causal_attention(*map(split, (q, k, v)))
        return y.transpose(1, 2).reshape(batch, time, channels)


class GQA(nn.Module):
    def __init__(self, d_model: int = 32, n_heads: int = 4, n_kv_heads: int = 2) -> None:
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.h, self.g, self.d = n_heads, n_kv_heads, d_model // n_heads
        self.q = nn.Linear(d_model, n_heads * self.d, bias=False)
        self.k = nn.Linear(d_model, n_kv_heads * self.d, bias=False)
        self.v = nn.Linear(d_model, n_kv_heads * self.d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        q = self.q(x).view(batch, time, self.h, self.d).transpose(1, 2)
        k = self.k(x).view(batch, time, self.g, self.d).transpose(1, 2)
        v = self.v(x).view(batch, time, self.g, self.d).transpose(1, 2)
        # 计算时重复，但生成时 cache 只存 g 组 K/V。
        repeat = self.h // self.g
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        y = causal_attention(q, k, v)
        return y.transpose(1, 2).reshape(batch, time, channels)


class ToyMLA(nn.Module):
    """压缩 K/V 成 latent，再恢复为 K/V 的可运行教学近似。"""

    def __init__(self, d_model: int = 32, n_heads: int = 4, latent_dim: int = 8) -> None:
        super().__init__()
        self.h, self.d, self.latent_dim = n_heads, d_model // n_heads, latent_dim
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv_compress = nn.Linear(d_model, latent_dim, bias=False)
        self.kv_restore = nn.Linear(latent_dim, 2 * d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        q = self.q(x).view(batch, time, self.h, self.d).transpose(1, 2)
        latent = self.kv_compress(x)                 # 可缓存：[B,T,latent_dim]
        k, v = self.kv_restore(latent).chunk(2, -1)
        split = lambda z: z.view(batch, time, self.h, self.d).transpose(1, 2)
        y = causal_attention(q, *map(split, (k, v)))
        return y.transpose(1, 2).reshape(batch, time, channels)


def main() -> None:
    torch.manual_seed(3)
    batch, time, d_model, heads, kv_heads, latent = 2, 8, 32, 4, 2, 8
    x = torch.randn(batch, time, d_model)
    for name, module, cache_elements in [
        ("MHA", MHA(d_model, heads), 2 * batch * time * heads * (d_model // heads)),
        ("GQA", GQA(d_model, heads, kv_heads), 2 * batch * time * kv_heads * (d_model // heads)),
        ("ToyMLA", ToyMLA(d_model, heads, latent), batch * time * latent),
    ]:
        y = module(x)
        assert y.shape == x.shape
        print(f"{name:>6}: output={tuple(y.shape)}, cache 教学估计={cache_elements} floats")


if __name__ == "__main__":
    main()
