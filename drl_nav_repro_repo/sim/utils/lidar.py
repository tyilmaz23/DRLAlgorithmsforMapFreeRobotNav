"""Simple 2D LiDAR ray casting."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from sim.utils.geometry import Rect, segment_intersect_aabb


def cast_lidar(
    x: float,
    y: float,
    theta: float,
    rects: Iterable[Rect],
    arena_min: float,
    arena_max: float,
    n_beams: int,
    max_range: float,
) -> np.ndarray:
    """Cast n_beams uniformly over 360 degrees.

    Walls are modeled as the arena square [arena_min, arena_max] × [arena_min, arena_max].

    Returns:
        ranges: (n_beams,) distances clipped to [0, max_range]
    """
    origin = np.array([x, y], dtype=np.float64)
    angles = theta + np.linspace(-np.pi, np.pi, n_beams, endpoint=False)

    ranges = np.full((n_beams,), max_range, dtype=np.float32)

    # Pre-build wall rectangles (thin AABBs)
    wall_thickness = 1e-3
    walls: List[Rect] = [
        Rect(cx=arena_min - wall_thickness / 2, cy=(arena_min + arena_max) / 2, w=wall_thickness, h=(arena_max - arena_min)),
        Rect(cx=arena_max + wall_thickness / 2, cy=(arena_min + arena_max) / 2, w=wall_thickness, h=(arena_max - arena_min)),
        Rect(cx=(arena_min + arena_max) / 2, cy=arena_min - wall_thickness / 2, w=(arena_max - arena_min), h=wall_thickness),
        Rect(cx=(arena_min + arena_max) / 2, cy=arena_max + wall_thickness / 2, w=(arena_max - arena_min), h=wall_thickness),
    ]

    all_rects = list(rects) + walls

    for i, a in enumerate(angles):
        d = np.array([np.cos(a), np.sin(a)], dtype=np.float64)
        best = max_range
        for r in all_rects:
            t = segment_intersect_aabb(origin, d, r)
            if t is None:
                continue
            if 0.0 <= t < best:
                best = t
        ranges[i] = np.float32(min(best, max_range))

    return ranges
