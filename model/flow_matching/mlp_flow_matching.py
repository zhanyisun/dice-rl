"""MLP models for flow matching policies."""

import torch
import torch.nn as nn

from model.common.mlp import MLP
from model.diffusion.modules import SinusoidalPosEmb


class TimeResidualBlock(nn.Module):
    """Residual block modulated by timestep embeddings."""

    def __init__(self, dim: int, hidden_dim: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        time_bias = self.time_mlp(time_emb)
        x = self.norm1(x)
        x = self.act(self.fc1(x + time_bias))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        x = x + residual
        x = self.norm2(x)
        return x


class FlowMatchingMLP(nn.Module):
    """MLP network for flow matching policy."""

    def __init__(
        self,
        action_dim: int,
        horizon_steps: int,
        cond_dim: int,
        time_dim: int = 16,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        cond_mlp_dims=None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        output_dim = action_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        if cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type="GELU",
                out_activation_type="Identity",
            )
            cond_out_dim = cond_mlp_dims[-1]
        else:
            cond_out_dim = cond_dim

        input_dim = time_dim + action_dim * horizon_steps + cond_out_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [TimeResidualBlock(hidden_dim, hidden_dim * 4, time_dim, dropout) for _ in range(num_blocks)]
        )
        self.final = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.final[-1].weight)
        nn.init.zeros_(self.final[-1].bias)
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: dict, **kwargs) -> torch.Tensor:
        B, Ta, Da = x.shape
        x = x.view(B, -1)
        state = cond["state"].view(B, -1)
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        x = torch.cat([x, time_emb, state], dim=-1)

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, time_emb)
        out = self.final(x)
        return out.view(B, Ta, Da)