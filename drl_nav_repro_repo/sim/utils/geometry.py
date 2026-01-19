"""Geometry utilities for 2D navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle."""

    cx: float
    cy: float
    w: float
    h: float

    @property
    def xmin(self) -> float:
        return self.cx - self.w / 2

    @property
    def xmax(self) -> float:
        return self.cx + self.w / 2

    @property
    def ymin(self) -> float:
        return self.cy - self.h / 2

    @property
    def ymax(self) -> float:
        return self.cy + self.h / 2

    def contains(self, x: float, y: float) -> bool:
        return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_to_pi(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def segment_intersect_aabb(ray_o: np.ndarray, ray_d: np.ndarray, rect: Rect) -> float | None:
    """Ray-AABB intersection distance (2D).

    Args:
        ray_o: origin (2,)
        ray_d: direction (2,), should be unit.
        rect: axis-aligned rectangle

    Returns:
        Distance t >= 0 to intersection, or None if no hit.
    """
    # Slab method
    inv = np.where(np.abs(ray_d) > 1e-9, 1.0 / ray_d, np.inf)
    t1 = (np.array([rect.xmin, rect.ymin]) - ray_o) * inv
    t2 = (np.array([rect.xmax, rect.ymax]) - ray_o) * inv

    tmin = np.maximum(np.minimum(t1, t2), 0.0)
    tmax = np.minimum(np.maximum(t1, t2), np.inf)

    t_enter = float(np.max(tmin))
    t_exit = float(np.min(tmax))

    if t_enter <= t_exit:
        return t_enter
    return None
