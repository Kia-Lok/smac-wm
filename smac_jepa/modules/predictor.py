from __future__ import annotations

import torch
from torch import nn

from smac_jepa.modules.blocks import AttentionBlock, MLP


class JEPAActionPredictor(nn.Module):
    """Predicts next latents from observation latents and action-history conditioning."""

    def __init__(
        self,
        latent_dim: int,
        n_agents: int,
        n_actions: int,
        action_dim: int,
        hidden_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.action_encoder = MLP(n_agents * n_actions, hidden_dim, action_dim)
        self.input_proj = nn.Linear(latent_dim + action_dim, latent_dim)
        self.block = AttentionBlock(latent_dim, num_heads=num_heads, hidden_dim=hidden_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, latents: torch.Tensor, conditioning_actions: torch.Tensor) -> torch.Tensor:
        batch, steps = latents.shape[:2]
        action_flat = conditioning_actions.reshape(batch, steps, self.n_agents * self.n_actions)
        action_emb = self.action_encoder(action_flat)
        x = torch.cat([latents, action_emb], dim=-1)
        x = self.input_proj(x)
        x = self.block(x)
        return self.output(x)
