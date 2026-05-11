#!/usr/bin/env bash
set -euo pipefail

python -m smac_jepa.evaluate \
  --data data/2s3z_random.npz \
  --checkpoint runs/2s3z_cpu/checkpoint.pt \
  --out runs/2s3z_cpu/eval_metrics.json

