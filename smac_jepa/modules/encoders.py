from __future__ import annotations

import torch
from torch import nn

from smac_jepa.modules.blocks import MLP


class StateEncoder(nn.Module):
    """Small vector-state encoder replacing LeWM's pixel encoder."""

    def __init__(self, state_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            MLP(state_dim, hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)

