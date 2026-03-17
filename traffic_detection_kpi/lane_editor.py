from __future__ import annotations

import math

import cv2
import numpy as np


def find_nearest_vertex(
    point: tuple[int, int], polygon: list[list[int]], threshold: float = 15.0
) -> int | None:
    """Return index of nearest vertex within threshold, or None."""
    px, py = point
    best_idx = None
    best_dist = threshold + 1
    for i, (vx, vy) in enumerate(polygon):
        dist = math.hypot(px - vx, py - vy)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx if best_dist <= threshold else None


def _point_to_segment_dist(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Distance from point (px,py) to line segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def find_nearest_edge(
    point: tuple[int, int], polygon: list[list[int]], threshold: float = 10.0
) -> int | None:
    """Return index of nearest edge within threshold, or None.
    Edge i connects polygon[i] to polygon[(i+1) % len(polygon)]."""
    px, py = point
    best_idx = None
    best_dist = threshold + 1
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        dist = _point_to_segment_dist(px, py, ax, ay, bx, by)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx if best_dist <= threshold else None


def project_point_on_edge(
    point: tuple[int, int], v1: list[int], v2: list[int]
) -> list[int]:
    """Project point onto edge v1-v2, return as [int, int]."""
    px, py = point
    ax, ay = v1
    bx, by = v2
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return [ax, ay]
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return [round(ax + t * dx), round(ay + t * dy)]


def can_delete_vertex(vertex_count: int) -> bool:
    """Return True if a vertex can be deleted (polygon keeps >= 3 vertices)."""
    return vertex_count > 3


class LaneEditor:
    def __init__(self, frame: np.ndarray, lanes: list[dict], lane_colors: list[tuple]) -> None:
        self._frame = frame
        self._lanes = lanes
        self._lane_colors = lane_colors

    def run(self) -> tuple[bool, list[dict]]:
        raise NotImplementedError("LaneEditor.run() not yet implemented")
