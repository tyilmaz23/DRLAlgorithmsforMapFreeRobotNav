"""Run a few episodes with a random policy to ensure the environment works."""

from __future__ import annotations

import argparse

import numpy as np

from scripts.common import load_yaml, make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--env", type=str, default="configs/env.yaml")
    ap.add_argument("--reward", type=str, default="configs/reward.yaml")
    args = ap.parse_args()

    env_cfg = load_yaml(args.env)
    reward_cfg = load_yaml(args.reward)
    env = make_env(env_cfg, reward_cfg, seed=args.seed, render=args.render)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        trunc = False
        ret = 0.0
        while not (done or trunc):
            a = env.action_space.sample()
            obs, r, done, trunc, info = env.step(a)
            ret += float(r)
        print(f"ep={ep} return={ret:.3f} success={info['success']} collision={info['collision']} timeout={info['timeout']}")


if __name__ == "__main__":
    main()
