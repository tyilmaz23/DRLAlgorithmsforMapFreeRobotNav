"""Gymnasium environment for map-free navigation with LiDAR-only perception.

This implementation mirrors the protocol described in the paper:
- 10m x 10m arena
- Episode-randomized static rectangular obstacles (axis-aligned)
- Differential-drive kinematics
- 2D LiDAR with configurable beams (20 for tuning, 120 for final)
- Observation scaled to [-1, 1] for tanh-based policies
- Reward = sparse terminal + dense shaping terms (progress, step, speed, proximity)

Important: If your original simulator differs in details, adjust `configs/env.yaml` and
this environment accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sim.utils.geometry import Rect, clamp, wrap_to_pi
from sim.utils.lidar import cast_lidar


@dataclass
class EnvConfig:
    arena_size: float = 10.0
    dt: float = 0.2
    max_steps: int = 1000

    # Robot limits (TurtleBot3 Burger)
    v_max: float = 0.22
    w_max: float = 2.84
    robot_radius: float = 0.105

    # Goal
    goal_radius: float = 0.25

    # Obstacles
    n_obs_min: int = 6
    n_obs_max: int = 14
    obs_w_min: float = 0.25
    obs_w_max: float = 1.20
    obs_h_min: float = 0.25
    obs_h_max: float = 1.20
    clearance: float = 0.30  # min distance between obstacles/robot/goal

    # LiDAR
    n_beams: int = 120
    lidar_range: float = 2.0

    # Normalization distance (upper bound for mapping to [-1,1])
    d_max: float = 14.2  # approx sqrt(10^2 + 10^2)

    # Reward weights (from the paper table)
    r_goal: float = 1.0
    r_collision: float = -1.0
    r_timeout: float = -0.6

    kappa_d: float = 6.0e-4
    kappa_s: float = 3.5e-4
    kappa_v: float = 2.2e-4
    kappa_w: float = 1.8e-4

    beta_dd: float = 4.0e-4
    beta_path: float = 5.0e-4  # paper states tuned in [1e-4, 1e-3]

    d_th: float = 0.25
    kappa_p: float = 3.0e-4


def _linmap(x: float, lo: float, hi: float) -> float:
    """Map x in [lo, hi] to [-1, 1]."""
    if hi <= lo:
        return 0.0
    x = clamp(x, lo, hi)
    return 2.0 * (x - lo) / (hi - lo) - 1.0


class TB3NavEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, cfg: EnvConfig | None = None, render_mode: str = "none"):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.render_mode = render_mode

        # Action: (v_cmd, w_cmd) in [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        obs_dim = 6 + self.cfg.n_beams
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.w = 0.0

        self.goal = np.zeros((2,), dtype=np.float64)
        self.obstacles: List[Rect] = []
        self.step_count = 0
        self.path_len = 0.0
        self.d_hist: List[float] = []
        self.d0 = 0.0

        self._rng = np.random.default_rng(0)

        # Optional renderer
        self._fig = None
        self._ax = None
        if self.render_mode == "human":
            import matplotlib.pyplot as plt

            self._plt = plt
            self._fig, self._ax = plt.subplots(figsize=(5, 5))

    def seed(self, seed: int | None = None) -> None:
        if seed is None:
            return
        self._rng = np.random.default_rng(int(seed))

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.step_count = 0
        self.path_len = 0.0
        self.d_hist = []

        # Arena bounds
        half = self.cfg.arena_size / 2.0
        self.arena_min = -half
        self.arena_max = half

        # Obstacles + start/goal
        self.obstacles = self._sample_obstacles()
        self.x, self.y, self.theta = self._sample_start_pose()
        self.goal = np.array(self._sample_goal(), dtype=np.float64)

        self.v = 0.0
        self.w = 0.0

        d = self._goal_dist()
        self.d0 = d
        self.d_hist = [d] * 5

        obs = self._get_obs()
        info = self._get_info(success=False, collision=False, timeout=False)
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1

        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)

        # Map actions -> velocities
        v_cmd = float((a[0] + 1.0) / 2.0 * self.cfg.v_max)  # [0, v_max]
        w_cmd = float(a[1] * self.cfg.w_max)  # [-w_max, w_max]

        self.v = v_cmd
        self.w = w_cmd

        # Integrate motion
        x_prev, y_prev = self.x, self.y
        self.theta = wrap_to_pi(self.theta + self.w * self.cfg.dt)
        self.x += self.v * np.cos(self.theta) * self.cfg.dt
        self.y += self.v * np.sin(self.theta) * self.cfg.dt
        self.path_len += float(np.hypot(self.x - x_prev, self.y - y_prev))

        # Collision / wall
        collision = self._check_collision()
        success = (self._goal_dist() <= self.cfg.goal_radius)
        timeout = (self.step_count >= self.cfg.max_steps)

        terminated = bool(success or collision)
        truncated = bool((not terminated) and timeout)

        reward = self._compute_reward(success=success, collision=collision, timeout=timeout)

        obs = self._get_obs()
        info = self._get_info(success=success, collision=collision, timeout=timeout)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_info(self, *, success: bool, collision: bool, timeout: bool) -> Dict:
        return {
            "success": bool(success),
            "collision": bool(collision),
            "timeout": bool(timeout),
            "step": int(self.step_count),
            "goal": self.goal.copy(),
            "pose": np.array([self.x, self.y, self.theta], dtype=np.float32),
            "path_len": float(self.path_len),
        }

    def _goal_dist(self) -> float:
        return float(np.hypot(self.goal[0] - self.x, self.goal[1] - self.y))

    def _check_collision(self) -> bool:
        # Wall
        if (
            (self.x - self.cfg.robot_radius) < self.arena_min
            or (self.x + self.cfg.robot_radius) > self.arena_max
            or (self.y - self.cfg.robot_radius) < self.arena_min
            or (self.y + self.cfg.robot_radius) > self.arena_max
        ):
            return True

        # Obstacles (circle vs rect using closest-point distance)
        for r in self.obstacles:
            cx = clamp(self.x, r.xmin, r.xmax)
            cy = clamp(self.y, r.ymin, r.ymax)
            if (self.x - cx) ** 2 + (self.y - cy) ** 2 <= (self.cfg.robot_radius ** 2):
                return True
        return False

    def _lidar(self) -> np.ndarray:
        return cast_lidar(
            x=self.x,
            y=self.y,
            theta=self.theta,
            rects=self.obstacles,
            arena_min=self.arena_min,
            arena_max=self.arena_max,
            n_beams=self.cfg.n_beams,
            max_range=self.cfg.lidar_range,
        )

    def _get_obs(self) -> np.ndarray:
        # Observation per paper: [sinθ, cosθ, v~, w~, d~, Δd~] + lidar~
        d = self._goal_dist()
        self.d_hist.append(d)
        if len(self.d_hist) > 5:
            self.d_hist = self.d_hist[-5:]

        d_prev = self.d_hist[0]
        dd = d_prev - d

        sin_t = np.sin(self.theta)
        cos_t = np.cos(self.theta)

        v_t = _linmap(self.v, 0.0, self.cfg.v_max)
        w_t = _linmap(self.w, -self.cfg.w_max, self.cfg.w_max)
        d_t = _linmap(d, 0.0, self.cfg.d_max)

        # Normalize Δd using d_max and clamp to a small range
        dd_t = np.clip(dd / self.cfg.d_max, -1.0, 1.0)

        lidar = self._lidar()
        lidar_n = _linmap_vec(lidar, 0.0, self.cfg.lidar_range)

        obs = np.concatenate(
            [
                np.array([sin_t, cos_t, v_t, w_t, d_t, dd_t], dtype=np.float32),
                lidar_n.astype(np.float32),
            ]
        )
        return obs

    def _compute_reward(self, *, success: bool, collision: bool, timeout: bool) -> float:
        # Terminal events
        if success:
            base = self.cfg.r_goal
        elif collision:
            base = self.cfg.r_collision
        elif timeout:
            base = self.cfg.r_timeout
        else:
            base = 0.0

        # Dense shaping
        d = self._goal_dist()
        d_prev = self.d_hist[-2] if len(self.d_hist) >= 2 else d

        progress = self.cfg.kappa_d * (d_prev - d)
        step_pen = -self.cfg.kappa_s
        v_pen = -self.cfg.kappa_v * abs(self.v)
        w_pen = -self.cfg.kappa_w * abs(self.w)

        lidar = self._lidar()
        dmin = float(np.min(lidar))
        prox = -self.cfg.kappa_p if dmin < self.cfg.d_th else 0.0

        # Incremental distance bonus (on normalized distance)
        d_prev_n = _linmap(d_prev, 0.0, self.cfg.d_max)
        d_n = _linmap(d, 0.0, self.cfg.d_max)
        inc = self.cfg.beta_dd * (d_prev_n - d_n)

        dense = progress + step_pen + v_pen + w_pen + prox + inc

        # Path-efficiency bonus at episode end (success or timeout)
        if success or timeout:
            if self.path_len > 1e-6:
                eff = (self.d0 - d) / self.path_len
                dense += self.cfg.beta_path * eff

        return float(base + dense)

    def _sample_obstacles(self) -> List[Rect]:
        n = int(self._rng.integers(self.cfg.n_obs_min, self.cfg.n_obs_max + 1))
        rects: List[Rect] = []
        tries = 0
        half = self.cfg.arena_size / 2.0

        while len(rects) < n and tries < 5000:
            tries += 1
            w = float(self._rng.uniform(self.cfg.obs_w_min, self.cfg.obs_w_max))
            h = float(self._rng.uniform(self.cfg.obs_h_min, self.cfg.obs_h_max))
            cx = float(self._rng.uniform(-half + w / 2, half - w / 2))
            cy = float(self._rng.uniform(-half + h / 2, half - h / 2))
            cand = Rect(cx=cx, cy=cy, w=w, h=h)

            # Check overlap with existing
            ok = True
            for r in rects:
                if _rects_too_close(cand, r, self.cfg.clearance):
                    ok = False
                    break
            if not ok:
                continue
            rects.append(cand)
        return rects

    def _sample_start_pose(self) -> Tuple[float, float, float]:
        # Sample a pose not in collision.
        half = self.cfg.arena_size / 2.0
        for _ in range(2000):
            x = float(self._rng.uniform(-half + 0.5, half - 0.5))
            y = float(self._rng.uniform(-half + 0.5, half - 0.5))
            th = float(self._rng.uniform(-np.pi, np.pi))
            if self._point_clear(x, y):
                return x, y, th
        return 0.0, 0.0, 0.0

    def _sample_goal(self) -> Tuple[float, float]:
        half = self.cfg.arena_size / 2.0
        for _ in range(2000):
            gx = float(self._rng.uniform(-half + 0.5, half - 0.5))
            gy = float(self._rng.uniform(-half + 0.5, half - 0.5))
            # Clear + far enough from start
            if (np.hypot(gx - self.x, gy - self.y) < 2.0):
                continue
            if self._point_clear(gx, gy):
                return gx, gy
        return float(self.x + 2.0), float(self.y)

    def _point_clear(self, x: float, y: float) -> bool:
        # Wall clearance
        if (
            x < self.arena_min + self.cfg.clearance
            or x > self.arena_max - self.cfg.clearance
            or y < self.arena_min + self.cfg.clearance
            or y > self.arena_max - self.cfg.clearance
        ):
            return False
        # Obstacle clearance
        for r in self.obstacles:
            cx = clamp(x, r.xmin - self.cfg.clearance, r.xmax + self.cfg.clearance)
            cy = clamp(y, r.ymin - self.cfg.clearance, r.ymax + self.cfg.clearance)
            if (x - cx) ** 2 + (y - cy) ** 2 <= (self.cfg.clearance ** 2):
                return False
        return True

    def render(self):
        if self.render_mode != "human":
            return
        ax = self._ax
        ax.clear()
        half = self.cfg.arena_size / 2.0
        ax.set_xlim([-half, half])
        ax.set_ylim([-half, half])
        ax.set_aspect("equal")

        # Walls
        ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], linewidth=2)

        # Obstacles
        for r in self.obstacles:
            ax.add_patch(self._plt.Rectangle((r.xmin, r.ymin), r.w, r.h, fill=True, alpha=0.4))

        # Goal
        ax.scatter([self.goal[0]], [self.goal[1]], marker="x", s=80)

        # Robot
        ax.scatter([self.x], [self.y], s=60)
        ax.arrow(self.x, self.y, 0.4 * np.cos(self.theta), 0.4 * np.sin(self.theta), head_width=0.12)

        ax.set_title(f"step={self.step_count}")
        self._fig.canvas.draw()
        self._plt.pause(0.001)


def _rects_too_close(a: Rect, b: Rect, clearance: float) -> bool:
    return not (
        (a.xmax + clearance) < (b.xmin - clearance)
        or (a.xmin - clearance) > (b.xmax + clearance)
        or (a.ymax + clearance) < (b.ymin - clearance)
        or (a.ymin - clearance) > (b.ymax + clearance)
    )


def _linmap_vec(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if hi <= lo:
        return np.zeros_like(arr)
    arr = np.clip(arr, lo, hi)
    return 2.0 * (arr - lo) / (hi - lo) - 1.0
