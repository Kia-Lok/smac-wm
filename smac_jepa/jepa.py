from __future__ import annotations

import torch
from torch import nn

from smac_jepa.modules import JEPAActionPredictor, StateEncoder, sigreg_loss


class SMACJEPA(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        n_actions: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        action_dim: int = 64,
        num_heads: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        self.encoder = StateEncoder(state_dim, hidden_dim, latent_dim)
        self.predictor = JEPAActionPredictor(
            latent_dim=latent_dim,
            n_agents=n_agents,
            n_actions=n_actions,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )

    def encode_state(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def predict_next(
        self,
        latents: torch.Tensor,
        conditioning_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.predictor(latents, conditioning_actions)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # observation sequence plus action-history conditioning predicts the
        # next observation sequence in latent space.
        latents = self.encode_state(batch["state_t"])
        target_latent = self.encode_state(batch["target_state"])
        pred_latent = self.predict_next(latents, batch["action_t"])
        return {
            "pred_latent": pred_latent,
            "target_latent": target_latent,
            "mask": batch["mask"],
        }

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        sigreg_weight: float = 0.01,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(batch)
        mask = out["mask"].unsqueeze(-1)
        denom = mask.sum().clamp_min(1.0) * out["pred_latent"].shape[-1]
        pred_loss = ((out["pred_latent"] - out["target_latent"]).pow(2) * mask).sum() / denom
        reg_loss = sigreg_loss(out["target_latent"], out["mask"])
        total = pred_loss + sigreg_weight * reg_loss
        return {
            "total_loss": total,
            "pred_loss": pred_loss,
            "sigreg_loss": reg_loss,
        }
