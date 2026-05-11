# SMAC-JEPA

CPU-first JEPA world model scaffold for SMACLite vector states.

## Install

Install the local dependencies, then install SMACLite from its repository or package source.

```bash
pip install -r requirements.txt
pip install git+https://github.com/uoe-agents/smaclite.git
```

## Collect Data

The collector runs a random valid-action policy and writes padded `.npz` trajectories.

```bash
python simulator/collect_smaclite_data.py \
  --env-key smaclite:smaclite/2s3z-v0 \
  --episodes 100 \
  --max-steps 120 \
  --out data/2s3z_random.npz \
  --seed 1
```

The saved file contains `states`, `actions`, `action_onehot`, `rewards`, `dones`,
`valid`, `avail_actions`, and scenario metadata.

## Train on CPU

```bash
python -m smac_jepa.train \
  --data data/2s3z_random.npz \
  --out-dir runs/2s3z_cpu \
  --epochs 5 \
  --batch-size 32
```

Training always uses CPU. Losses are written to:

- `runs/2s3z_cpu/loss_log.csv`
- `runs/2s3z_cpu/loss_log.jsonl`
- `runs/2s3z_cpu/checkpoint.pt`
- `runs/2s3z_cpu/config.json`

## Evaluate

```bash
python -m smac_jepa.evaluate \
  --data data/2s3z_random.npz \
  --checkpoint runs/2s3z_cpu/checkpoint.pt \
  --out runs/2s3z_cpu/eval_metrics.json
```

Stage one reports next-state embedding MSE. A useful first signal is finite
training loss that trends downward over a few epochs on collected trajectories.

## Synthetic Smoke Test

If SMACLite is not installed yet, create a tiny synthetic dataset and run the
same CPU training path:

```bash
python scripts/smoke_synthetic.py --out data/synthetic.npz
python -m smac_jepa.train --data data/synthetic.npz --out-dir runs/synthetic_cpu --epochs 2
python -m smac_jepa.evaluate \
  --data data/synthetic.npz \
  --checkpoint runs/synthetic_cpu/checkpoint.pt \
  --out runs/synthetic_cpu/eval_metrics.json
```
