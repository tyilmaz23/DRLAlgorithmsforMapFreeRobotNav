"""Shared helpers for training/evaluation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

from sim.envs.tb3_nav_env import EnvConfig, TB3NavEnv


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(env_cfg: Dict[str, Any], reward_cfg: Dict[str, Any], seed: int, render: bool = False) -> TB3NavEnv:
    cfg = EnvConfig(**{**env_cfg, **reward_cfg})
    env = TB3NavEnv(cfg=cfg, render_mode="human" if render else "none")
    env.seed(seed)
    return env


def cosine_lr(initial_lr: float, min_lr_ratio: float = 0.05):
    """Cosine annealing schedule compatible with SB3.

    SB3 passes `progress_remaining` in [1, 0].
    """

    def _fn(progress_remaining: float) -> float:
        # convert to t in [0,1]
        t = 1.0 - float(progress_remaining)
        import math

        lr = (min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * t))) * initial_lr
        return float(lr)

    return _fn


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
