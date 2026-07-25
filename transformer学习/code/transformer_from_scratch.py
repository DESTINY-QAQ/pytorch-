"""可运行的教学版 decoder-only Transformer。

展示标准 MHA、GQA 与 Top-K MoE 的核心数据流，并训练一个字符级 next-token 模型。
它刻意使用小尺寸和直白写法，不是生产级训练代码。
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCharTokenizer:
    """教学用字符级 tokenizer；真实项目请换为 BPE/SentencePiece。"""

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class MultiHeadSelfAttention(nn.Module):
    """标准因果多头自注意力（MHA）。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))  # each [B, H, T, D]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,T,T]
        causal = torch.tril(torch.ones(time, time, device=x.device, dtype=torch.bool))
        weights = F.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.proj(y)


class GroupedQueryAttention(nn.Module):
    """教学版 GQA：H 个 Q 头，G 个共享的 K/V 头。"""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        q = self.q_proj(x).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # 同一组 K/V 服务多个 Q 头。真实 KV cache 只需存 n_kv_heads 组。
        repeat = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.tril(torch.ones(time, time, device=x.device, dtype=torch.bool))
        weights = F.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        y = weights @ v
        return self.out_proj(y.transpose(1, 2).contiguous().view(batch, time, channels))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, multiplier: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, multiplier * d_model),
            nn.GELU(),
            nn.Linear(multiplier * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopKMoE(nn.Module):
    """最小 token 级 Top-K MoE。为清晰使用循环，不适合大规模训练。"""

    def __init__(self, d_model: int, n_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(d_model) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        flat = x.reshape(batch * time, channels)
        probs = F.softmax(self.router(flat), dim=-1)
        weights, expert_ids = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            selected = (expert_ids == expert_index).any(dim=-1)
            if selected.any():
                selected_weight = (weights[selected] * (expert_ids[selected] == expert_index)).sum(dim=-1)
                output[selected] += expert(flat[selected]) * selected_weight.unsqueeze(-1)
        return output.view(batch, time, channels)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


@dataclass
class Config:
    vocab_size: int
    block_size: int = 32
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.05


class MiniTransformer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # 常见的权重共享

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, time = token_ids.shape
        assert time <= self.cfg.block_size, "序列超过 block_size"
        positions = torch.arange(time, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))  # [B, T, V]

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, new_tokens: int, temperature: float = 0.9) -> torch.Tensor:
        self.eval()
        for _ in range(new_tokens):
            logits = self(ids[:, -self.cfg.block_size:])[:, -1, :] / temperature
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


TEXT = ("Transformer 学习从 token 开始。token 变成向量，注意力读取上下文，"
        "模型预测下一个 token。小模型也能跑通完整训练循环。\n") * 80


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def smoke_test() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 5, 32)
    assert MultiHeadSelfAttention(32, 4)(x).shape == x.shape
    assert GroupedQueryAttention(32, 4, 2)(x).shape == x.shape
    assert TopKMoE(32, n_experts=4, top_k=2)(x).shape == x.shape
    model = MiniTransformer(Config(vocab_size=20, block_size=8, d_model=32, n_heads=4, n_layers=2))
    assert model(torch.randint(0, 20, (2, 8))).shape == (2, 8, 20)
    print("all smoke tests passed")


def train_demo(steps: int) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = TinyCharTokenizer(TEXT)
    data = torch.tensor(tokenizer.encode(TEXT), dtype=torch.long)
    cfg = Config(vocab_size=tokenizer.vocab_size)
    model = MiniTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print(f"device={device}, vocab={cfg.vocab_size}, parameters={sum(p.numel() for p in model.parameters()):,}")
    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch(data, 16, cfg.block_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0:
            print(f"step {step:>4}/{steps}: loss={loss.item():.4f}")
    seed = torch.tensor([tokenizer.encode("Transformer ")], device=device)
    print("\n--- sample ---")
    print(tokenizer.decode(model.generate(seed, 120)[0].tolist()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="只检查模块形状")
    parser.add_argument("--train", action="store_true", help="训练字符级演示模型")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
    if args.train:
        train_demo(args.steps)
    if not args.smoke and not args.train:
        parser.print_help()


if __name__ == "__main__":
    main()
