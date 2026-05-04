from dataclasses import dataclass

import torch
from torch import nn

from ..components import FactoryConfig


@dataclass
class Relative3DBiasConfig(FactoryConfig):
    num_heads: int = 6
    hidden_dim: int = 64
    use_distance: bool = True

    @property
    def CLS(self) -> type["Relative3DBias"]:
        return Relative3DBias

    def instantiate(self) -> "Relative3DBias":
        return Relative3DBias(
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            use_distance=self.use_distance,
        )


class Relative3DBias(nn.Module):
    def __init__(
        self,
        num_heads: int = 6,
        hidden_dim: int = 64,
        use_distance: bool = True,
    ) -> None:
        super().__init__()
        self.use_distance = use_distance
        in_dim = 4 if use_distance else 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads, bias=False),
        )

    def forward(self, q_centers: torch.Tensor, k_centers: torch.Tensor) -> torch.Tensor:
        # q_centers: (B, Q, 3), k_centers: (B, K, 3)
        rel = q_centers.unsqueeze(2) - k_centers.unsqueeze(1)  # (B, Q, K, 3)
        if self.use_distance:
            dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
            feats = torch.cat((rel, dist), dim=-1)  # (B, Q, K, 4)
        else:
            feats = rel
        bias = self.mlp(feats)  # (B, Q, K, num_heads)
        return bias.permute(0, 3, 1, 2).contiguous()  # (B, num_heads, Q, K)
