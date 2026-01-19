# Map-Free DRL Navigation (Sim-to-Real) — Reproducibility Repository

This repository contains the training/evaluation pipeline used in the IEEE Access paper:

**"A Comparative Study of DRL Algorithms for Map-Free Robot Navigation with Zero-Shot Sim-to-Real Transfer"**

It provides:
- A **custom Python simulator** (Gymnasium environment) with **episode-randomized static obstacles** in a 10m×10m arena.
- Training scripts for **PPO, SAC, A2C, TD3, DDPG** (Stable-Baselines3).
- **Optuna** hyperparameter optimization scripts (**80 trials**, **3e5 steps/trial**) with a fixed-seed protocol.
- Evaluation scripts computing success/collision/timeout rates and path-length statistics.
- Documentation for **ROS/Gazebo** and **TurtleBot3 Burger** deployment (zero-shot transfer).

> Note: If you are releasing this repository for peer review, please replace all placeholder strings like `GITHUB_REPO_URL` with your actual repository link, and optionally include your trained checkpoints under `checkpoints/`.

## Quickstart

### 1) Environment setup

**Option A (pip):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Option B (conda):**
```bash
conda env create -f environment.yml
conda activate drl_nav
```

### 2) Sanity check (run a random policy)
```bash
python scripts/sanity_check_env.py --episodes 3
```

### 3) Train (Stage-2, full budget: 100,000 episodes)
```bash
python scripts/train.py --algo PPO --episodes 100000 --seed 42 --cfg configs/best_hparams.yaml
```

### 4) Evaluate (deterministic)
```bash
python scripts/eval.py --algo PPO --model runs/PPO_seed42/model.zip --episodes 20 --seed 42 --deterministic
```

### 5) Hyperparameter tuning (Optuna)
```bash
python scripts/optuna_tune.py --algo PPO --trials 80 --steps-per-trial 300000 --seed 123 \
  --search-space configs/optuna_search_space.yaml
```

## Reproducibility protocol (as in the paper)

- **Arena:** 10m×10m, 4 walls.
- **Obstacles:** axis-aligned rectangles; randomized at the start of each episode; stationary within the episode.
- **Control interval:** Δt = 0.2 s.
- **Max episode length:** 1000 steps.
- **Two-stage budget allocation:**
  1) Stage-1 feasibility screen: **10,000 episodes** for all algorithms.
  2) Stage-2 retrain from scratch: **100,000 episodes** for selected algorithms.
- **Seed averaging:** mean over **5 independent seeds**.

See `docs/REPRODUCE_PAPER.md`.

## Repository layout
- `sim/`            : custom simulator (Gymnasium env + LiDAR ray casting)
- `scripts/`        : training/evaluation/optuna + helpers
- `configs/`        : environment, reward, hyperparameters
- `docs/`           : reproduction + Gazebo/TurtleBot3 notes
- `ros_gazebo/`     : placeholders for ROS2/Gazebo integration
- `checkpoints/`    : optional released model artifacts

## Citation
If you use this code, please cite the paper.

## License
Choose a license (e.g., MIT) and place it in `LICENSE`.
