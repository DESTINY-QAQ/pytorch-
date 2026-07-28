"""Embedding 训练的最小 CBOW 演示。

运行：
    conda activate tf
    python code/embedding_training_demo.py --steps 400

它把“中心词周围的词”作为输入，预测中心词。训练后可查看若干词的余弦相似度。
这是教学玩具：语料很小，输出只用于观察 API 和训练流程，不代表真实词义空间。
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


SENTENCES = [
    "猫 喜欢 吃 鱼",
    "小猫 喜欢 吃 鱼",
    "狗 喜欢 吃 骨头",
    "小狗 喜欢 吃 骨头",
    "猫 和 狗 都 是 动物",
] * 80


def build_examples(token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """把左右各一个词作为 context，预测中间的 center word。"""
    contexts, targets = [], []
    for i in range(1, len(token_ids) - 1):
        contexts.append([token_ids[i - 1], token_ids[i + 1]])
        targets.append(token_ids[i])
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class CBOW(nn.Module):
    """context IDs -> 平均 embedding -> 词表 logits。"""

    def __init__(self, vocab_size: int, dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        # context_ids: [B, 2]
        context_vectors = self.embedding(context_ids)  # [B, 2, dim]
        mean_vector = context_vectors.mean(dim=1)      # [B, dim]
        return self.output(mean_vector)                # [B, vocab_size]


def main(steps: int) -> None:
    torch.manual_seed(7)
    tokens = " ".join(SENTENCES).split()
    vocab = sorted(set(tokens))
    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for word, i in stoi.items()}
    ids = [stoi[word] for word in tokens]
    contexts, targets = build_examples(ids)

    model = CBOW(len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)
    for step in range(1, steps + 1):
        logits = model(contexts)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, steps // 5) == 0:
            print(f"step {step:>4}/{steps}: loss={loss.item():.4f}")

    with torch.no_grad():
        vectors = F.normalize(model.embedding.weight, dim=-1)
        print("\n--- 余弦相似度（仅观察小语料的趋势）---")
        for word in ("猫", "狗", "小猫"):
            if word not in stoi:
                continue
            scores = vectors @ vectors[stoi[word]]
            nearest = scores.topk(min(4, len(vocab))).indices.tolist()
            print(f"{word:>3} -> " + ", ".join(f"{itos[i]} ({scores[i]:.2f})" for i in nearest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    main(parser.parse_args().steps)
