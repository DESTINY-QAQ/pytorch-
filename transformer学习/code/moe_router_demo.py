"""Top-K MoE 路由、加权混合与负载均衡的最小演示。

运行：
    conda activate tf
    python code/moe_router_demo.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ToyMoE(nn.Module):
    def __init__(self, dim: int = 8, n_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B,T,C] -> 为了讲清楚，先压为 [tokens,C]
        flat = x.reshape(-1, x.size(-1))
        probabilities = F.softmax(self.router(flat), dim=-1)
        top_weights, top_ids = probabilities.topk(self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            selected = (top_ids == expert_id).any(dim=-1)
            if selected.any():
                # 某 token 若此专家被选中，取它对应的 Top-K 权重；否则为 0。
                weight = (top_weights[selected] * (top_ids[selected] == expert_id)).sum(dim=-1)
                output[selected] += expert(flat[selected]) * weight.unsqueeze(-1)
        return output.view_as(x), probabilities, top_ids


def main() -> None:
    torch.manual_seed(5)
    x = torch.randn(2, 3, 8)  # 6 个 token，每个 8 维
    model = ToyMoE()
    y, probabilities, top_ids = model(x)
    print("output shape:", tuple(y.shape))
    print("\n每个 token 的路由概率：")
    for index, row in enumerate(probabilities):
        print(f"token {index}: " + " ".join(f"E{i}={p:.2f}" for i, p in enumerate(row.tolist())))
        print("         Top-2 experts:", top_ids[index].tolist())

    # 辅助损失的直观版本：希望所有专家收到的 token 比例更均匀。
    load = torch.bincount(top_ids.flatten(), minlength=probabilities.size(-1)).float()
    load = load / load.sum()
    balance_loss = ((load - 1 / probabilities.size(-1)) ** 2).mean()
    print("\n专家负载比例:", [round(v, 3) for v in load.tolist()])
    print("教学版负载均衡损失:", round(balance_loss.item(), 6))


if __name__ == "__main__":
    main()
