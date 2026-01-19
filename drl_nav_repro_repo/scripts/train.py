"""Train a DRL agent in the custom Python simulator.

Example:
  python scripts/train.py --algo PPO --episodes 100000 --seed 42 --cfg configs/best_hparams.yaml

This script follows the paper protocol:
- max_steps=1000 per episode
- dt=0.2s
- stage-2 full budget: 100,000 episodes (use --episodes)

Note: SB3 trains by environment steps, not episodes. We therefore map:
  total_timesteps = episodes * max_steps

If your original code used early termination or variable episode lengths, this mapping remains valid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor

from scripts.common import cosine_lr, ensure_dir, load_yaml, make_env, set_global_seeds


ALGOS = {
    "PPO": PPO,
    "SAC": SAC,
    "A2C": A2C,
    "TD3": TD3,
    "DDPG": DDPG,
}


def build_policy_kwargs(algo: str):
    if algo in ("PPO", "A2C"):
        # paper: pi=[256,256,128], vf=[128,128]
        return dict(net_arch=dict(pi=[256, 256, 128], vf=[128, 128]))
    if algo in ("SAC", "TD3", "DDPG"):
        # paper: pi=[256,256], qf=[256,256]
        return dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=sorted(ALGOS.keys()))
    ap.add_argument("--episodes", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=str, default="configs/best_hparams.yaml")
    ap.add_argument("--env", type=str, default="configs/env.yaml")
    ap.add_argument("--reward", type=str, default="configs/reward.yaml")
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    set_global_seeds(args.seed)

    hparams_all = load_yaml(args.cfg)
    env_cfg = load_yaml(args.env)
    reward_cfg = load_yaml(args.reward)

    hparams = dict(hparams_all.get(args.algo, {}))

    # Build env
    env = make_env(env_cfg, reward_cfg, seed=args.seed, render=args.render)
    env = Monitor(env)

    max_steps = int(env_cfg.get("max_steps", 1000))
    total_timesteps = int(args.episodes * max_steps)

    # Learning rate schedule
    lr = float(hparams.get("learning_rate", 3e-4))
    lr_sched = hparams.get("lr_schedule", "constant")
    if lr_sched == "cosine":
        learning_rate = cosine_lr(lr)
    else:
        learning_rate = lr

    policy_kwargs = build_policy_kwargs(args.algo)

    # SB3 common args
    common = dict(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
    )

    # Algo-specific args
    if args.algo == "PPO":
        model = PPO(
            **common,
            batch_size=int(hparams.get("batch_size", 256)),
            gamma=float(hparams.get("gamma", 0.99)),
            n_steps=int(hparams.get("n_steps", 1024)),
            clip_range=float(hparams.get("clip_range", 0.2)),
            gae_lambda=float(hparams.get("gae_lambda", 0.95)),
        )
    elif args.algo == "A2C":
        model = A2C(
            **common,
            gamma=float(hparams.get("gamma", 0.99)),
            n_steps=int(hparams.get("n_steps", 5)),
            gae_lambda=float(hparams.get("gae_lambda", 0.95)),
            ent_coef=float(hparams.get("ent_coef", 0.0)),
            vf_coef=float(hparams.get("vf_coef", 0.5)),
            max_grad_norm=float(hparams.get("max_grad_norm", 0.5)),
        )
    elif args.algo == "SAC":
        model = SAC(
            **common,
            batch_size=int(hparams.get("batch_size", 256)),
            gamma=float(hparams.get("gamma", 0.99)),
            tau=float(hparams.get("tau", 0.005)),
            train_freq=int(hparams.get("train_freq", 1)),
            gradient_steps=int(hparams.get("gradient_steps", 1)),
            ent_coef=float(hparams.get("ent_coef", "auto")) if isinstance(hparams.get("ent_coef", "auto"), str) else float(hparams.get("ent_coef", 0.0)),
            buffer_size=int(hparams.get("buffer_size", 100000)),
            learning_starts=int(hparams.get("learning_starts", 1000)),
            target_update_interval=int(hparams.get("target_update_interval", 1)),
        )
    elif args.algo == "TD3":
        model = TD3(
            **common,
            batch_size=int(hparams.get("batch_size", 256)),
            gamma=float(hparams.get("gamma", 0.99)),
            tau=float(hparams.get("tau", 0.005)),
            train_freq=int(hparams.get("train_freq", 1)),
            gradient_steps=int(hparams.get("gradient_steps", 1)),
            policy_delay=int(hparams.get("policy_delay", 2)),
        )
    elif args.algo == "DDPG":
        model = DDPG(
            **common,
            batch_size=int(hparams.get("batch_size", 256)),
            gamma=float(hparams.get("gamma", 0.99)),
            tau=float(hparams.get("tau", 0.005)),
            train_freq=int(hparams.get("train_freq", 1)),
            gradient_steps=int(hparams.get("gradient_steps", 1)),
        )
    else:
        raise ValueError(f"Unsupported algo: {args.algo}")

    run_dir = ensure_dir(Path(args.outdir) / f"{args.algo}_seed{args.seed}")
    model_path = run_dir / "model.zip"

    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(str(model_path))

    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
