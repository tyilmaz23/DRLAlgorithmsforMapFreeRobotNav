"""Evaluate a trained SB3 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from scripts.common import load_yaml, make_env


ALGOS = {
    "PPO": PPO,
    "SAC": SAC,
    "A2C": A2C,
    "TD3": TD3,
    "DDPG": DDPG,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=sorted(ALGOS.keys()))
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--env", type=str, default="configs/env.yaml")
    ap.add_argument("--reward", type=str, default="configs/reward.yaml")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    env_cfg = load_yaml(args.env)
    reward_cfg = load_yaml(args.reward)
    env = make_env(env_cfg, reward_cfg, seed=args.seed)

    ModelCls = ALGOS[args.algo]
    model = ModelCls.load(args.model)

    succ = 0
    coll = 0
    tout = 0
    lengths = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        trunc = False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, trunc, info = env.step(action)

        succ += int(info.get("success", False))
        coll += int(info.get("collision", False))
        tout += int(info.get("timeout", False))
        if info.get("success", False):
            lengths.append(float(info.get("path_len", 0.0)))

    n = float(args.episodes)
    print("=== Evaluation ===")
    print(f"episodes       : {args.episodes}")
    print(f"success_rate   : {succ / n:.4f}")
    print(f"collision_rate : {coll / n:.4f}")
    print(f"timeout_rate   : {tout / n:.4f}")
    if lengths:
        print(f"avg_path_len   : {np.mean(lengths):.3f} m")
    else:
        print("avg_path_len   : n/a")


if __name__ == "__main__":
    main()
