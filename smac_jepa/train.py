from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smac_jepa.config import TrainConfig
from smac_jepa.data import SMACJEPADataset
from smac_jepa.jepa import SMACJEPA
from smac_jepa.utils import set_seed
from smac_jepa.utils.logging import LossLogger
from smac_jepa.utils.plots import write_svg_line_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMAC-JEPA on CPU")
    parser.add_argument("--data", nargs="+", required=True, help="One or more NPZ datasets")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--sigreg-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    config = TrainConfig(**vars(args))
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save(out_dir / "config.json")

    set_seed(config.seed)
    device = torch.device("cpu")
    dataset = SMACJEPADataset(config.data, context_len=config.context_len)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    model = SMACJEPA(
        state_dim=dataset.metadata.state_dim,
        n_agents=dataset.metadata.n_agents,
        n_actions=dataset.metadata.n_actions,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        action_dim=config.action_dim,
        num_heads=config.num_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    logger = LossLogger(out_dir, "loss_log")
    epoch_logger = LossLogger(out_dir, "epoch_loss")

    global_step = 0
    step_rows: list[dict[str, float | int]] = []
    epoch_rows: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(1, config.epochs + 1):
        epoch_sums = {"total_loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0}
        epoch_batches = 0
        for batch in loader:
            global_step += 1
            epoch_batches += 1
            batch = to_device(batch, device)
            losses = model.loss(batch, sigreg_weight=config.sigreg_weight)
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            optimizer.step()

            row = {
                "epoch": epoch,
                "step": global_step,
                "total_loss": float(losses["total_loss"].detach().cpu()),
                "pred_loss": float(losses["pred_loss"].detach().cpu()),
                "sigreg_loss": float(losses["sigreg_loss"].detach().cpu()),
            }
            logger.log(row)
            step_rows.append(row)
            for key in epoch_sums:
                epoch_sums[key] += row[key]
            if global_step == 1 or global_step % config.log_every == 0:
                print(
                    "epoch={epoch} step={step} total_loss={total_loss:.6f} "
                    "pred_loss={pred_loss:.6f} sigreg_loss={sigreg_loss:.6f}".format(**row),
                    flush=True,
                )
        epoch_row = {
            "epoch": epoch,
            "step": global_step,
            "total_loss": epoch_sums["total_loss"] / max(epoch_batches, 1),
            "pred_loss": epoch_sums["pred_loss"] / max(epoch_batches, 1),
            "sigreg_loss": epoch_sums["sigreg_loss"] / max(epoch_batches, 1),
        }
        epoch_logger.log(epoch_row)
        epoch_rows.append(epoch_row)
        print(
            "epoch_summary epoch={epoch} step={step} total_loss={total_loss:.6f} "
            "pred_loss={pred_loss:.6f} sigreg_loss={sigreg_loss:.6f}".format(**epoch_row),
            flush=True,
        )

    write_svg_line_plot(
        epoch_rows,
        "epoch",
        "total_loss",
        "Average Total Loss Per Epoch",
        out_dir / "loss_by_epoch.svg",
    )
    write_svg_line_plot(
        epoch_rows,
        "epoch",
        "pred_loss",
        "Average Prediction Loss Per Epoch",
        out_dir / "pred_loss_by_epoch.svg",
    )
    write_svg_line_plot(
        step_rows,
        "step",
        "pred_loss",
        "Prediction Loss Per Training Step",
        out_dir / "pred_loss_by_step.svg",
    )
    print(
        "wrote_plots "
        f"{out_dir / 'loss_by_epoch.svg'} "
        f"{out_dir / 'pred_loss_by_epoch.svg'} "
        f"{out_dir / 'pred_loss_by_step.svg'}",
        flush=True,
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "metadata": {
                "state_dim": dataset.metadata.state_dim,
                "n_agents": dataset.metadata.n_agents,
                "n_actions": dataset.metadata.n_actions,
            },
            "config": vars(args),
        },
        out_dir / "checkpoint.pt",
    )


if __name__ == "__main__":
    main()
