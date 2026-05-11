from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic SMAC-JEPA NPZ")
    parser.add_argument("--out", default="data/synthetic.npz")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=10)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    states = np.zeros(
        (args.episodes, args.steps + 1, args.state_dim), dtype=np.float32
    )
    actions = rng.integers(
        0,
        args.n_actions,
        size=(args.episodes, args.steps, args.n_agents),
        dtype=np.int64,
    )
    action_onehot = np.eye(args.n_actions, dtype=np.float32)[actions]
    rewards = rng.normal(size=(args.episodes, args.steps)).astype(np.float32)
    dones = np.zeros((args.episodes, args.steps), dtype=bool)
    valid = np.ones((args.episodes, args.steps), dtype=bool)
    avail_actions = np.ones(
        (args.episodes, args.steps, args.n_agents, args.n_actions), dtype=np.float32
    )

    dynamics = rng.normal(
        scale=0.05,
        size=(args.n_agents * args.n_actions, args.state_dim),
    ).astype(np.float32)
    states[:, 0] = rng.normal(size=(args.episodes, args.state_dim))
    for step in range(args.steps):
        action_features = action_onehot[:, step].reshape(args.episodes, -1)
        states[:, step + 1] = (
            states[:, step]
            + action_features @ dynamics
            + rng.normal(scale=0.01, size=(args.episodes, args.state_dim))
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        states=states,
        actions=actions,
        action_onehot=action_onehot,
        rewards=rewards,
        dones=dones,
        valid=valid,
        avail_actions=avail_actions,
        scenario=np.asarray("synthetic"),
        state_dim=np.asarray(args.state_dim, dtype=np.int64),
        n_agents=np.asarray(args.n_agents, dtype=np.int64),
        n_actions=np.asarray(args.n_actions, dtype=np.int64),
        max_steps=np.asarray(args.steps, dtype=np.int64),
    )
    print(f"Saved synthetic dataset to {out}")


if __name__ == "__main__":
    main()

