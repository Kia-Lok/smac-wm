#!/usr/bin/env bash
set -euo pipefail

python -m smac_jepa.train \
  --data data/2s3z_random.npz \
  --out-dir runs/2s3z_cpu \
  --epochs 5 \
  --batch-size 32 \
  --latent-dim 64 \
  --hidden-dim 128 \
  --context-len 1

