from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetMetadata:
    state_dim: int
    n_agents: int
    n_actions: int


def _as_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def load_npz_metadata(path: str | Path) -> DatasetMetadata:
    with np.load(path, allow_pickle=False) as data:
        return DatasetMetadata(
            state_dim=int(data["state_dim"]),
            n_agents=int(data["n_agents"]),
            n_actions=int(data["n_actions"]),
        )


def _one_hot(actions: np.ndarray, n_actions: int) -> np.ndarray:
    clipped = np.clip(actions.astype(np.int64), 0, n_actions - 1)
    return np.eye(n_actions, dtype=np.float32)[clipped]


class SMACJEPADataset(Dataset):
    """Windowed JEPA dataset.

    Each item contains an observation sequence, an action-history conditioning
    sequence, and the next observation sequence shifted by one step.
    """

    def __init__(self, paths: str | Path | Iterable[str | Path], context_len: int = 1):
        if context_len < 1:
            raise ValueError("context_len must be at least 1")
        self.paths = _as_paths(paths)
        self.context_len = context_len
        if not self.paths:
            raise ValueError("At least one dataset path is required")

        states_list: list[np.ndarray] = []
        actions_list: list[np.ndarray] = []
        valid_list: list[np.ndarray] = []
        metadata: DatasetMetadata | None = None

        for path in self.paths:
            with np.load(path, allow_pickle=False) as data:
                current = DatasetMetadata(
                    state_dim=int(data["state_dim"]),
                    n_agents=int(data["n_agents"]),
                    n_actions=int(data["n_actions"]),
                )
                if metadata is None:
                    metadata = current
                elif metadata != current:
                    raise ValueError(
                        f"Dataset metadata mismatch in {path}: {current} != {metadata}"
                    )

                states = data["states"].astype(np.float32)
                if "action_onehot" in data:
                    actions = data["action_onehot"].astype(np.float32)
                else:
                    actions = _one_hot(data["actions"], current.n_actions)
                if "valid" in data:
                    valid = data["valid"].astype(bool)
                else:
                    valid = np.ones(actions.shape[:2], dtype=bool)

                states_list.append(states)
                actions_list.append(actions)
                valid_list.append(valid)

        assert metadata is not None
        self.metadata = metadata
        self.states = np.concatenate(states_list, axis=0)
        self.actions = np.concatenate(actions_list, axis=0)
        self.valid = np.concatenate(valid_list, axis=0)
        self.index: list[tuple[int, int]] = []

        episode_count, horizon = self.actions.shape[:2]
        for episode_idx in range(episode_count):
            for start in range(0, horizon - context_len + 1):
                window_valid = self.valid[episode_idx, start : start + context_len]
                if np.all(window_valid):
                    self.index.append((episode_idx, start))

        if not self.index:
            raise ValueError("No valid training windows found")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_idx, start = self.index[idx]
        end = start + self.context_len
        return {
            # Observation sequence: state_t ... state_{t+K-1}
            "state_t": torch.from_numpy(self.states[episode_idx, start:end]),
            # Conditioning sequence: actions taken from those previous states.
            "action_t": torch.from_numpy(self.actions[episode_idx, start:end]),
            # Prediction target sequence: state_{t+1} ... state_{t+K}
            "target_state": torch.from_numpy(self.states[episode_idx, start + 1 : end + 1]),
            "mask": torch.from_numpy(self.valid[episode_idx, start:end].astype(np.float32)),
        }
