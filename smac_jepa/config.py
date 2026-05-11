from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class TrainConfig:
    data: list[str]
    out_dir: str
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    latent_dim: int = 64
    hidden_dim: int = 128
    action_dim: int = 64
    context_len: int = 1
    num_heads: int = 2
    sigreg_weight: float = 0.01
    seed: int = 1
    num_workers: int = 0
    log_every: int = 10

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

