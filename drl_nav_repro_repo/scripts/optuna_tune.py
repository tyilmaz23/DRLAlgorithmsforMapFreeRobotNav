"""Optuna hyperparameter tuning.

Protocol (paper):
- 80 trials per algorithm
- 3e5 environment steps per trial
- objective: mean success rate over 10 deterministic eval episodes, averaged over the last 5 evaluation checkpoints
- fixed seed for sampler; Hyperband pruning

This script provides a clean, review-friendly implementation of that protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import yaml

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from scripts.common import cosine_lr, ensure_dir, load_yaml, make_env, set_global_seeds


ALGOS = {
    "PPO": PPO,
    "SAC": SAC,
    "A2C": A2C,
    "TD3": TD3,
    "DDPG": DDPG,
}


def suggest_from_space(trial: optuna.Trial, space: Dict[str, Any]) -> Any:
    t = space["type"]
    if t == "categorical":
        return trial.suggest_categorical(space.get("name"), space["choices"])  # name overwritten by caller
    if t == "uniform":
        return trial.suggest_float(space.get("name"), float(space["low"]), float(space["high"]))
    if t == "loguniform":
        return trial.suggest_float(space.get("name"), float(space["low"]), float(space["high"]), log=True)
    raise ValueError(f"Unknown space type: {t}")


def sample_hparams(trial: optuna.Trial, algo: str, search: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # common
    for k, spec in search.get("common", {}).items():
        spec = dict(spec)
        spec["name"] = k
        out[k] = suggest(trial, spec)

    # algo-specific
    for k, spec in search.get(algo, {}).items():
        spec = dict(spec)
        spec["name"] = k
        out[k] = suggest(trial, spec)

    return out


def suggest(trial: optuna.Trial, spec: Dict[str, Any]) -> Any:
    name = spec["name"]
    t = spec["type"]
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if t == "uniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if t == "loguniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    raise ValueError(f"Unknown space type: {t}")


def build_policy_kwargs(algo: str) -> Dict[str, Any]:
    if algo in ("PPO", "A2C"):
        return dict(net_arch=dict(pi=[256, 256, 128], vf=[128, 128]))
    if algo in ("SAC", "TD3", "DDPG"):
        return dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    return {}


def evaluate_success_rate(env, model, n_eval: int, deterministic: bool, seed0: int) -> float:
    succ = 0
    for i in range(n_eval):
        obs, info = env.reset(seed=seed0 + i)
        done = False
        trunc = False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, trunc, info = env.step(action)
        succ += int(info.get("success", False))
    return float(succ) / float(n_eval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=sorted(ALGOS.keys()))
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--steps-per-trial", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--search-space", type=str, default="configs/optuna_search_space.yaml")
    ap.add_argument("--env", type=str, default="configs/env.yaml")
    ap.add_argument("--reward", type=str, default="configs/reward.yaml")
    ap.add_argument("--outdir", type=str, default="runs/optuna")
    args = ap.parse_args()

    set_global_seeds(args.seed)

    search = load_yaml(args.search_space)
    env_cfg = load_yaml(args.env)
    reward_cfg = load_yaml(args.reward)

    outdir = ensure_dir(Path(args.outdir) / args.algo)

    # Pruner + Sampler (paper mentions Hyperband + TPE with fixed seed)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.HyperbandPruner()

    def objective(trial: optuna.Trial) -> float:
        # sample hparams
        hp = sample_hparams(trial, args.algo, search)

        # learning rate schedule (cosine) is used for PPO in the paper's final config.
        lr = float(hp.get("learning_rate", 3e-4))
        learning_rate = cosine_lr(lr) if args.algo == "PPO" else lr

        env = make_env(env_cfg, reward_cfg, seed=args.seed)
        ModelCls = ALGOS[args.algo]
        policy_kwargs = build_policy_kwargs(args.algo)

        common = dict(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            seed=args.seed,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
        )

        if args.algo == "PPO":
            model = ModelCls(
                **common,
                batch_size=int(hp.get("batch_size", 256)),
                gamma=float(hp.get("gamma", 0.99)),
                n_steps=int(hp.get("n_steps", 1024)),
                clip_range=float(hp.get("clip_range", 0.2)),
                gae_lambda=float(hp.get("gae_lambda", 0.95)),
            )
        elif args.algo == "A2C":
            model = ModelCls(
                **common,
                gamma=float(hp.get("gamma", 0.99)),
                n_steps=int(hp.get("n_steps", 10)),
                gae_lambda=float(hp.get("gae_lambda", 0.95)),
                ent_coef=float(hp.get("ent_coef", 0.0)),
                vf_coef=float(hp.get("vf_coef", 0.5)),
            )
        elif args.algo == "SAC":
            model = ModelCls(
                **common,
                batch_size=int(hp.get("batch_size", 256)),
                gamma=float(hp.get("gamma", 0.99)),
                tau=float(hp.get("tau", 0.005)),
                train_freq=int(hp.get("train_freq", 1)),
                gradient_steps=int(hp.get("gradient_steps", 1)),
                ent_coef=float(hp.get("ent_coef", 0.0)),
                buffer_size=int(hp.get("buffer_size", 100000)),
            )
        elif args.algo == "TD3":
            model = ModelCls(
                **common,
                batch_size=int(hp.get("batch_size", 256)),
                gamma=float(hp.get("gamma", 0.99)),
                tau=float(hp.get("tau", 0.005)),
                train_freq=int(hp.get("train_freq", 1)),
                gradient_steps=int(hp.get("gradient_steps", 1)),
                policy_delay=int(hp.get("policy_delay", 2)),
            )
        elif args.algo == "DDPG":
            model = ModelCls(
                **common,
                batch_size=int(hp.get("batch_size", 256)),
                gamma=float(hp.get("gamma", 0.99)),
                tau=float(hp.get("tau", 0.005)),
                train_freq=int(hp.get("train_freq", 1)),
                gradient_steps=int(hp.get("gradient_steps", 1)),
            )
        else:
            raise ValueError(args.algo)

        # Evaluation checkpoints: 5 equally spaced checkpoints
        n_checkpoints = 5
        eval_every = max(1, args.steps_per_trial // n_checkpoints)

        scores = []
        steps_done = 0

        for ck in range(n_checkpoints):
            model.learn(total_timesteps=eval_every, reset_num_timesteps=False)
            steps_done += eval_every

            score = evaluate_success_rate(env, model, n_eval=10, deterministic=True, seed0=args.seed + 1000 + ck * 100)
            scores.append(score)

            # report intermediate result for pruning
            trial.report(float(np.mean(scores)), step=ck)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # objective: mean over last 5 evaluation checkpoints (here exactly 5)
        return float(np.mean(scores[-5:]))

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=args.trials)

    best = {
        "algo": args.algo,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }

    out_path = outdir / "best_result.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best, f, sort_keys=False)

    print(f"Saved Optuna result to: {out_path}")


if __name__ == "__main__":
    main()
