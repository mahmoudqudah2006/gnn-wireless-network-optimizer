from __future__ import annotations

import torch
from torch import nn


def normalized_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device)
    a_hat = adjacency + identity
    degree = a_hat.sum(dim=1)
    inv_sqrt = torch.pow(degree.clamp_min(1e-12), -0.5)
    return inv_sqrt[:, None] * a_hat * inv_sqrt[None, :]


class GCN(nn.Module):
    def __init__(self, in_features: int = 3, hidden: int = 24) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden)
        self.layer2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        norm = normalized_adjacency(adjacency)
        hidden = torch.relu(self.layer1(norm @ x))
        return self.layer2(norm @ hidden).squeeze(-1)
