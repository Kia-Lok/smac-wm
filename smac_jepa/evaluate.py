from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smac_jepa.data import SMACJEPADataset
from smac_jepa.jepa import SMACJEPA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SMAC-JEPA next-state embedding loss")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    dataset = SMACJEPADataset(args.data, context_len=int(config["context_len"]))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    metadata = checkpoint["metadata"]
    model = SMACJEPA(
        state_dim=metadata["state_dim"],
        n_agents=metadata["n_agents"],
        n_actions=metadata["n_actions"],
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        action_dim=int(config["action_dim"]),
        num_heads=int(config["num_heads"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    total_loss = 0.0
    total_count = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            mask = out["mask"].unsqueeze(-1)
            squared = (out["pred_latent"] - out["target_latent"]).pow(2) * mask
            total_loss += float(squared.sum().cpu())
            total_count += float(mask.sum().cpu()) * out["pred_latent"].shape[-1]

    metrics = {
        "next_state_embedding_mse": total_loss / max(total_count, 1.0),
        "num_windows": len(dataset),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

