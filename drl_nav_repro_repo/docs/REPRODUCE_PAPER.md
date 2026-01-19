# Reproduce the paper protocol

This document provides a concise, reviewer-friendly checklist for reproducing the experiments.

## 0) Install
See `README.md`.

## 1) Environment protocol
- Arena: 10m × 10m with rigid walls.
- Obstacles: axis-aligned rectangles randomized per episode (size, placement). Obstacles are stationary within the episode.
- Control frequency: Δt = 0.2 s.
- Max episode length: 1000 steps.
- Observation (final):
  - `[sin(theta), cos(theta), v~, w~, d~, Δd~]` concatenated with `120` LiDAR beams within 2.0 m.
  - All components are scaled to `[-1, 1]`.

The default values are in `configs/env.yaml` and `configs/reward.yaml`.

## 2) Stage-1 feasibility screen (10,000 episodes)
Run for each algorithm:
```bash
python scripts/train.py --algo PPO --episodes 10000 --seed 42 --cfg configs/best_hparams.yaml
python scripts/train.py --algo SAC --episodes 10000 --seed 42 --cfg configs/best_hparams.yaml
python scripts/train.py --algo A2C --episodes 10000 --seed 42 --cfg configs/best_hparams.yaml
python scripts/train.py --algo TD3 --episodes 10000 --seed 42 --cfg configs/best_hparams.yaml
python scripts/train.py --algo DDPG --episodes 10000 --seed 42 --cfg configs/best_hparams.yaml
```

## 3) Stage-2 full retraining (100,000 episodes, 5 seeds)
Example for PPO:
```bash
for s in 42 43 44 45 46; do
  python scripts/train.py --algo PPO --episodes 100000 --seed $s --cfg configs/best_hparams.yaml
done
```

## 4) Deterministic evaluation
```bash
python scripts/eval.py --algo PPO --model runs/PPO_seed42/model.zip --episodes 20 --seed 42 --deterministic
```

## 5) Optuna tuning protocol
The paper uses a fixed-budget tuning protocol:
- 80 trials per algorithm
- 3×10^5 environment steps per trial
- objective = mean success rate across 10 deterministic eval episodes, averaged over the last 5 evaluation checkpoints

Run:
```bash
python scripts/optuna_tune.py --algo PPO --trials 80 --steps-per-trial 300000 --seed 123 \
  --search-space configs/optuna_search_space.yaml
```
