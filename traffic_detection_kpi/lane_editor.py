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


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_STATUS_HEIGHT = 40
_VERTEX_RADIUS = 6
_SELECTED_RADIUS = 8


class LaneEditor:
    def __init__(self, frame: np.ndarray, lanes: list[dict], lane_colors: list[tuple]) -> None:
        self._frame = frame.copy()
        self._h, self._w = frame.shape[:2]
        self._lanes = [dict(l) for l in lanes]
        for lane in self._lanes:
            lane["polygon"] = [list(p) for p in lane["polygon"]]
        # Clamp any out-of-frame vertices into the frame
        for lane in self._lanes:
            lane["polygon"] = [self._clamp(p[0], p[1]) for p in lane["polygon"]]
        self._lane_colors = list(lane_colors)
        self._original_lanes = [
            {"name": l["name"], "polygon": [list(p) for p in l["polygon"]]}
            for l in self._lanes
        ]

        self._modified = False
        self._draw_mode = False
        self._draw_points: list[list[int]] = []

        self._sel_lane_idx: int | None = None
        self._sel_vert_idx: int | None = None
        self._dragging = False
        self._active_lane_idx: int | None = None

        self._window_name = "Lane Editor"

    def _clamp(self, x: int, y: int) -> list[int]:
        return [max(0, min(x, self._w - 1)), max(0, min(y, self._h - 1))]

    def run(self) -> tuple[bool, list[dict]]:
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self._window_name, self._mouse_cb)
        self._redraw()

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == 255:  # no key pressed
                continue
            if key == ord("q"):
                if self._draw_mode:
                    self._draw_mode = False
                    self._draw_points.clear()
                    self._redraw()
                result = self._handle_quit()
                if result is not None:
                    cv2.destroyWindow(self._window_name)
                    return result
            elif key == ord("n"):
                self._draw_mode = True
                self._draw_points.clear()
                self._sel_lane_idx = None
                self._sel_vert_idx = None
                self._redraw()
            elif key == ord("d"):
                self._delete_active_lane()
            elif key == 27:  # Esc
                self._draw_mode = False
                self._draw_points.clear()
                self._sel_lane_idx = None
                self._sel_vert_idx = None
                self._active_lane_idx = None
                self._redraw()
            elif key == 13:  # Enter
                if self._draw_mode and len(self._draw_points) >= 3:
                    self._finish_new_lane()

        cv2.destroyWindow(self._window_name)
        return False, self._original_lanes

    def _mouse_cb(self, event, x, y, flags, param):
        if self._draw_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._draw_points.append(self._clamp(x, y))
                self._redraw()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            for li, lane in enumerate(self._lanes):
                vi = find_nearest_vertex((x, y), lane["polygon"], threshold=15)
                if vi is not None:
                    self._sel_lane_idx = li
                    self._sel_vert_idx = vi
                    self._dragging = True
                    self._redraw()
                    return

            for li, lane in enumerate(self._lanes):
                ei = find_nearest_edge((x, y), lane["polygon"], threshold=10)
                if ei is not None:
                    new_pt = project_point_on_edge(
                        (x, y), lane["polygon"][ei],
                        lane["polygon"][(ei + 1) % len(lane["polygon"])]
                    )
                    lane["polygon"].insert(ei + 1, new_pt)
                    self._sel_lane_idx = li
                    self._sel_vert_idx = ei + 1
                    self._modified = True
                    self._redraw()
                    return

            self._sel_lane_idx = None
            self._sel_vert_idx = None
            self._redraw()

        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            if self._sel_lane_idx is not None and self._sel_vert_idx is not None:
                self._lanes[self._sel_lane_idx]["polygon"][self._sel_vert_idx] = self._clamp(x, y)
                self._modified = True
                self._redraw()

        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            # First check: right-click inside a polygon selects it for deletion with 'd'
            for li, lane in enumerate(self._lanes):
                poly = np.array(lane["polygon"], dtype=np.int32)
                if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                    self._active_lane_idx = li
                    self._redraw()
                    return

            # Second check: right-click near a vertex (outside any polygon) deletes it
            for li, lane in enumerate(self._lanes):
                vi = find_nearest_vertex((x, y), lane["polygon"], threshold=15)
                if vi is not None:
                    if can_delete_vertex(len(lane["polygon"])):
                        lane["polygon"].pop(vi)
                        self._modified = True
                        self._sel_lane_idx = None
                        self._sel_vert_idx = None
                        self._redraw()
                    return

    def _delete_active_lane(self):
        if self._active_lane_idx is not None and self._active_lane_idx < len(self._lanes):
            self._lanes.pop(self._active_lane_idx)
            self._lane_colors.pop(self._active_lane_idx)
            self._active_lane_idx = None
            self._sel_lane_idx = None
            self._sel_vert_idx = None
            self._modified = True
            self._redraw()

    def _finish_new_lane(self):
        cv2.destroyWindow(self._window_name)
        name = input("Enter lane name: ").strip()
        if not name:
            name = f"Lane {len(self._lanes) + 1}"
        from traffic_detection_kpi.annotator import LANE_PALETTE
        color = LANE_PALETTE[len(self._lanes) % len(LANE_PALETTE)]
        self._lanes.append({"name": name, "polygon": list(self._draw_points)})
        self._lane_colors.append(color)
        self._draw_mode = False
        self._draw_points.clear()
        self._modified = True
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_cb)
        self._redraw()

    def _handle_quit(self) -> tuple[bool, list[dict]] | None:
        if not self._modified:
            return False, self._original_lanes
        while True:
            ans = input("Save changes? (y/n/c): ").strip().lower()
            if ans == "y":
                return True, self._lanes
            elif ans == "n":
                return False, self._original_lanes
            elif ans == "c":
                return None

    def _redraw(self):
        canvas = self._frame.copy()

        overlay = canvas.copy()
        for i, lane in enumerate(self._lanes):
            poly = np.array(lane["polygon"], dtype=np.int32)
            color = self._lane_colors[i] if i < len(self._lane_colors) else (128, 128, 128)
            cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        for i, lane in enumerate(self._lanes):
            poly = np.array(lane["polygon"], dtype=np.int32)
            color = self._lane_colors[i] if i < len(self._lane_colors) else (128, 128, 128)
            thickness = 3 if i == self._active_lane_idx else 2
            cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=thickness)

            for vi, (vx, vy) in enumerate(lane["polygon"]):
                if i == self._sel_lane_idx and vi == self._sel_vert_idx:
                    cv2.circle(canvas, (vx, vy), _SELECTED_RADIUS, color, -1)
                else:
                    cv2.circle(canvas, (vx, vy), _VERTEX_RADIUS, _WHITE, 1)

            pts = lane["polygon"]
            cx = sum(p[0] for p in pts) // len(pts)
            cy = sum(p[1] for p in pts) // len(pts)
            cv2.putText(canvas, lane["name"], (cx - 20, cy), _FONT, 0.5, color, 1, cv2.LINE_AA)

        if self._draw_mode and self._draw_points:
            for pt in self._draw_points:
                cv2.circle(canvas, tuple(pt), _VERTEX_RADIUS, _WHITE, -1)
            if len(self._draw_points) > 1:
                pts = np.array(self._draw_points, dtype=np.int32)
                cv2.polylines(canvas, [pts], isClosed=False, color=_WHITE, thickness=2)

        h, w = canvas.shape[:2]
        cv2.rectangle(canvas, (0, h - _STATUS_HEIGHT), (w, h), (30, 30, 30), -1)
        if self._draw_mode:
            status = f"DRAW - click to place points ({len(self._draw_points)} placed), Enter to finish, Esc to cancel"
        else:
            status = "EDIT - n: new lane | d: delete lane | q: quit | Esc: deselect"
        cv2.putText(canvas, status, (10, h - 12), _FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(self._window_name, canvas)
