# Lane Editor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone interactive lane polygon editor (`traffic-lane-editor`) using OpenCV, with save-back to YAML config.

**Architecture:** New `LaneEditor` class in `lane_editor.py` handles all OpenCV mouse/keyboard interaction on a static video frame. New `editor_cli.py` provides the CLI entry point, loads config, launches editor, and saves changes. Hit detection math is exposed as module-level functions for testability.

**Tech Stack:** OpenCV (display + mouse callbacks), PyYAML (config read/write), numpy

**Spec:** `docs/superpowers/specs/2026-03-17-lane-editor-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `traffic_detection_kpi/annotator.py` | Modify | Rename `_PALETTE` to `LANE_PALETTE` (public) |
| `traffic_detection_kpi/lane_editor.py` | Create | `LaneEditor` class — display, mouse callbacks, keyboard handling, hit detection |
| `traffic_detection_kpi/editor_cli.py` | Create | CLI entry point + `save_lanes_to_config()` |
| `pyproject.toml` | Modify | Add `traffic-lane-editor` script entry point |
| `tests/test_lane_editor.py` | Create | Unit tests for hit detection, config saving |

---

## Chunk 1: Palette Rename + Config Save + Tests

### Task 1: Rename _PALETTE to LANE_PALETTE

**Files:**
- Modify: `traffic_detection_kpi/annotator.py`

- [ ] **Step 1: Rename _PALETTE to LANE_PALETTE in annotator.py**

In `traffic_detection_kpi/annotator.py`, replace all occurrences of `_PALETTE` with `LANE_PALETTE`.

Lines to change:
- Line 9: `_PALETTE = [` → `LANE_PALETTE = [`
- Line 35: `self._lane_colors = [_PALETTE[i % len(_PALETTE)]` → `self._lane_colors = [LANE_PALETTE[i % len(LANE_PALETTE)]`

- [ ] **Step 2: Run existing annotator tests**

Run: `PYTHONPATH=. pytest tests/test_annotator.py -v`
Expected: All 4 tests PASS (rename is internal, tests access `_lane_colors`)

- [ ] **Step 3: Commit**

```bash
git add traffic_detection_kpi/annotator.py
git commit -m "refactor: make LANE_PALETTE public in annotator for reuse"
```

### Task 2: Config save function — tests

**Files:**
- Create: `tests/test_lane_editor.py`

- [ ] **Step 4: Write config save tests**

```python
import math
import yaml
import pytest


def test_save_lanes_to_config_roundtrip(tmp_path):
    """Save modified lanes, reload, verify lanes match."""
    from traffic_detection_kpi.editor_cli import save_lanes_to_config
    from traffic_detection_kpi.config import load_config

    config_content = """\
video_path: "test.mp4"
output_dir: "./output"
model:
  path: "yolo11m.pt"
  confidence: 0.2
  classes: [car]
tracker:
  type: deepsort
  max_age: 20
  n_init: 2
  max_cosine_distance: 0.8
  embedder: mobilenet
lanes:
  - name: "Lane 1"
    polygon: [[0, 0], [100, 0], [100, 100], [0, 100]]
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    new_lanes = [
        {"name": "Lane A", "polygon": [[10, 10], [200, 10], [200, 200], [10, 200]]},
        {"name": "Lane B", "polygon": [[300, 300], [400, 300], [400, 400]]},
    ]
    save_lanes_to_config(str(config_path), new_lanes)

    config = load_config(str(config_path))
    assert len(config.lanes) == 2
    assert config.lanes[0].name == "Lane A"
    assert config.lanes[0].polygon == [[10, 10], [200, 10], [200, 200], [10, 200]]
    assert config.lanes[1].name == "Lane B"
    assert config.lanes[1].polygon == [[300, 300], [400, 300], [400, 400]]


def test_save_lanes_preserves_other_config(tmp_path):
    """Non-lane config fields survive save."""
    from traffic_detection_kpi.editor_cli import save_lanes_to_config

    config_content = """\
video_path: "test.mp4"
output_dir: "./output"
model:
  path: "yolo11m.pt"
  confidence: 0.2
  classes: [car, truck]
tracker:
  type: deepsort
  max_age: 20
  n_init: 2
  max_cosine_distance: 0.8
  embedder: mobilenet
lanes:
  - name: "Old"
    polygon: [[0, 0], [1, 0], [1, 1]]
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    save_lanes_to_config(str(config_path), [
        {"name": "New", "polygon": [[5, 5], [10, 5], [10, 10]]},
    ])

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    assert raw["video_path"] == "test.mp4"
    assert raw["output_dir"] == "./output"
    assert raw["model"]["path"] == "yolo11m.pt"
    assert raw["model"]["classes"] == ["car", "truck"]
    assert raw["tracker"]["type"] == "deepsort"
    assert raw["tracker"]["max_age"] == 20
    assert len(raw["lanes"]) == 1
    assert raw["lanes"][0]["name"] == "New"
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_lane_editor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'traffic_detection_kpi.editor_cli'`

### Task 3: Config save function — implementation

**Files:**
- Create: `traffic_detection_kpi/editor_cli.py`

- [ ] **Step 6: Implement save_lanes_to_config and CLI stub**

```python
import argparse
import sys

import cv2
import yaml

from traffic_detection_kpi.annotator import LANE_PALETTE
from traffic_detection_kpi.config import load_config
from traffic_detection_kpi.lane_editor import LaneEditor


def save_lanes_to_config(config_path: str, lanes: list[dict]) -> None:
    """Overwrite the lanes section of a YAML config file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw["lanes"] = lanes
    with open(config_path, "w") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive lane polygon editor"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--video", help="Path to video file (overrides config video_path)")
    args = parser.parse_args()

    config = load_config(args.config)
    video_path = args.video or config.video_path
    if not video_path:
        print("Error: no video source. Use --video or set video_path in config.", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: cannot read first frame from video", file=sys.stderr)
        sys.exit(1)

    lanes = [{"name": l.name, "polygon": l.polygon} for l in config.lanes]
    colors = [LANE_PALETTE[i % len(LANE_PALETTE)] for i in range(len(lanes))]

    modified, updated_lanes = LaneEditor(frame, lanes, colors).run()

    if modified:
        save_lanes_to_config(args.config, updated_lanes)
        print(f"Saved {len(updated_lanes)} lanes to {args.config}")
    else:
        print("No changes saved.")


if __name__ == "__main__":
    main()
```

Note: This imports `LaneEditor` which doesn't exist yet. The config save tests only test `save_lanes_to_config` so they won't hit that import at test time since they import the function directly. However, the module-level import will fail. To fix this, we need to create a minimal `lane_editor.py` stub first.

- [ ] **Step 7: Create minimal lane_editor.py stub**

```python
from __future__ import annotations

import cv2
import numpy as np


class LaneEditor:
    def __init__(self, frame: np.ndarray, lanes: list[dict], lane_colors: list[tuple]) -> None:
        self._frame = frame
        self._lanes = lanes
        self._lane_colors = lane_colors

    def run(self) -> tuple[bool, list[dict]]:
        raise NotImplementedError("LaneEditor.run() not yet implemented")
```

- [ ] **Step 8: Run config save tests**

Run: `PYTHONPATH=. pytest tests/test_lane_editor.py -v -k "save"`
Expected: Both tests PASS

- [ ] **Step 9: Commit**

```bash
git add traffic_detection_kpi/editor_cli.py traffic_detection_kpi/lane_editor.py tests/test_lane_editor.py
git commit -m "feat: add editor CLI with config save function"
```

---

## Chunk 2: Hit Detection Math

### Task 4: Hit detection — tests

**Files:**
- Modify: `tests/test_lane_editor.py`

- [ ] **Step 10: Write hit detection tests**

Append to `tests/test_lane_editor.py`:

```python
def test_find_nearest_vertex_within_threshold():
    from traffic_detection_kpi.lane_editor import find_nearest_vertex

    polygon = [[100, 100], [200, 100], [200, 200], [100, 200]]
    # Click right on vertex 0
    result = find_nearest_vertex((100, 100), polygon, threshold=15)
    assert result == 0

    # Click 10px away from vertex 1
    result = find_nearest_vertex((210, 100), polygon, threshold=15)
    assert result == 1

    # Click too far from any vertex
    result = find_nearest_vertex((150, 150), polygon, threshold=15)
    assert result is None


def test_find_nearest_edge_within_threshold():
    from traffic_detection_kpi.lane_editor import find_nearest_edge

    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    # Click near edge 0 (top edge, y=0), at (50, 5)
    result = find_nearest_edge((50, 5), polygon, threshold=10)
    assert result == 0  # edge between vertex 0 and 1

    # Click near edge 2 (bottom edge, y=100), at (50, 97)
    result = find_nearest_edge((50, 97), polygon, threshold=10)
    assert result == 2

    # Click in center, far from edges
    result = find_nearest_edge((50, 50), polygon, threshold=10)
    assert result is None


def test_project_point_on_edge_returns_int():
    from traffic_detection_kpi.lane_editor import project_point_on_edge

    # Project (50, 5) onto edge (0,0)-(100,0) — should be (50, 0)
    result = project_point_on_edge((50, 5), [0, 0], [100, 0])
    assert result == [50, 0]
    assert isinstance(result[0], int)
    assert isinstance(result[1], int)

    # Project onto diagonal edge — result must be int
    result = project_point_on_edge((30, 25), [0, 0], [100, 100])
    assert isinstance(result[0], int)
    assert isinstance(result[1], int)


def test_no_delete_below_three_vertices():
    from traffic_detection_kpi.lane_editor import can_delete_vertex

    assert can_delete_vertex(4) is True   # 4 vertices -> can delete
    assert can_delete_vertex(3) is False  # 3 vertices -> minimum
    assert can_delete_vertex(2) is False  # should never happen but handle
```

- [ ] **Step 11: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_lane_editor.py -v -k "vertex or edge or project or delete"`
Expected: FAIL — `ImportError: cannot import name 'find_nearest_vertex'`

### Task 5: Hit detection — implementation

**Files:**
- Modify: `traffic_detection_kpi/lane_editor.py`

- [ ] **Step 12: Add hit detection functions to lane_editor.py**

Add these module-level functions before the `LaneEditor` class:

```python
import math


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

    Edge i connects polygon[i] to polygon[(i+1) % len(polygon)].
    """
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
```

- [ ] **Step 13: Run hit detection tests**

Run: `PYTHONPATH=. pytest tests/test_lane_editor.py -v`
Expected: All 6 tests PASS

- [ ] **Step 14: Run all tests for regressions**

Run: `PYTHONPATH=. pytest tests/ -v -k "not integration"`
Expected: All tests PASS

- [ ] **Step 15: Commit**

```bash
git add traffic_detection_kpi/lane_editor.py tests/test_lane_editor.py
git commit -m "feat: add hit detection math for lane editor"
```

---

## Chunk 3: LaneEditor Interactive Loop

### Task 6: LaneEditor full implementation

**Files:**
- Modify: `traffic_detection_kpi/lane_editor.py`

- [ ] **Step 16: Implement the full LaneEditor class**

Replace the `LaneEditor` class stub with the full implementation:

```python
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_STATUS_HEIGHT = 40
_VERTEX_RADIUS = 6
_SELECTED_RADIUS = 8


class LaneEditor:
    def __init__(self, frame: np.ndarray, lanes: list[dict], lane_colors: list[tuple]) -> None:
        self._frame = frame.copy()
        self._lanes = [dict(l) for l in lanes]  # deep-ish copy
        for lane in self._lanes:
            lane["polygon"] = [list(p) for p in lane["polygon"]]
        self._lane_colors = list(lane_colors)
        self._original_lanes = [
            {"name": l["name"], "polygon": [list(p) for p in l["polygon"]]}
            for l in self._lanes
        ]

        self._modified = False
        self._draw_mode = False
        self._draw_points: list[list[int]] = []

        # Selection state
        self._sel_lane_idx: int | None = None  # vertex drag selection
        self._sel_vert_idx: int | None = None
        self._dragging = False
        self._active_lane_idx: int | None = None  # right-click lane for deletion

        self._window_name = "Lane Editor"

    def run(self) -> tuple[bool, list[dict]]:
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_cb)
        self._redraw()

        while True:
            key = cv2.waitKey(30) & 0xFF
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
                self._draw_points.append([x, y])
                self._redraw()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Try vertex selection first
            for li, lane in enumerate(self._lanes):
                vi = find_nearest_vertex((x, y), lane["polygon"], threshold=15)
                if vi is not None:
                    self._sel_lane_idx = li
                    self._sel_vert_idx = vi
                    self._dragging = True
                    self._redraw()
                    return

            # Try edge insertion
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

            # Deselect
            self._sel_lane_idx = None
            self._sel_vert_idx = None
            self._redraw()

        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            if self._sel_lane_idx is not None and self._sel_vert_idx is not None:
                self._lanes[self._sel_lane_idx]["polygon"][self._sel_vert_idx] = [x, y]
                self._modified = True
                self._redraw()

        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Try vertex delete
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

            # Try lane selection for deletion
            for li, lane in enumerate(self._lanes):
                poly = np.array(lane["polygon"], dtype=np.int32)
                if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                    self._active_lane_idx = li
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
                return None  # cancel quit

    def _redraw(self):
        canvas = self._frame.copy()

        # Draw lane fills
        overlay = canvas.copy()
        for i, lane in enumerate(self._lanes):
            poly = np.array(lane["polygon"], dtype=np.int32)
            color = self._lane_colors[i] if i < len(self._lane_colors) else (128, 128, 128)
            cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        # Draw lane borders and vertices
        for i, lane in enumerate(self._lanes):
            poly = np.array(lane["polygon"], dtype=np.int32)
            color = self._lane_colors[i] if i < len(self._lane_colors) else (128, 128, 128)
            thickness = 3 if i == self._active_lane_idx else 2
            cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=thickness)

            # Draw vertices
            for vi, (vx, vy) in enumerate(lane["polygon"]):
                if i == self._sel_lane_idx and vi == self._sel_vert_idx:
                    cv2.circle(canvas, (vx, vy), _SELECTED_RADIUS, color, -1)
                else:
                    cv2.circle(canvas, (vx, vy), _VERTEX_RADIUS, _WHITE, 1)

            # Lane name at centroid
            pts = lane["polygon"]
            cx = sum(p[0] for p in pts) // len(pts)
            cy = sum(p[1] for p in pts) // len(pts)
            cv2.putText(canvas, lane["name"], (cx - 20, cy), _FONT, 0.5, color, 1, cv2.LINE_AA)

        # Draw in-progress polygon
        if self._draw_mode and self._draw_points:
            for pt in self._draw_points:
                cv2.circle(canvas, tuple(pt), _VERTEX_RADIUS, _WHITE, -1)
            if len(self._draw_points) > 1:
                pts = np.array(self._draw_points, dtype=np.int32)
                cv2.polylines(canvas, [pts], isClosed=False, color=_WHITE, thickness=2)

        # Status bar
        h, w = canvas.shape[:2]
        cv2.rectangle(canvas, (0, h - _STATUS_HEIGHT), (w, h), (30, 30, 30), -1)
        if self._draw_mode:
            status = f"DRAW — click to place points ({len(self._draw_points)} placed), Enter to finish, Esc to cancel"
        else:
            status = "EDIT — n: new lane | d: delete lane | q: quit | Esc: deselect"
        cv2.putText(canvas, status, (10, h - 12), _FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(self._window_name, canvas)
```

- [ ] **Step 17: Run all tests**

Run: `PYTHONPATH=. pytest tests/ -v -k "not integration"`
Expected: All tests PASS

- [ ] **Step 18: Commit**

```bash
git add traffic_detection_kpi/lane_editor.py
git commit -m "feat: implement LaneEditor with interactive polygon editing"
```

---

## Chunk 4: pyproject.toml + Final Verification

### Task 7: Add script entry point

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 19: Add traffic-lane-editor entry point**

In `pyproject.toml`, change the `[project.scripts]` section from:

```toml
[project.scripts]
traffic-kpi = "traffic_detection_kpi.__main__:main"
```

to:

```toml
[project.scripts]
traffic-kpi = "traffic_detection_kpi.__main__:main"
traffic-lane-editor = "traffic_detection_kpi.editor_cli:main"
```

- [ ] **Step 20: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add traffic-lane-editor CLI entry point"
```

### Task 8: Final verification

- [ ] **Step 21: Run full test suite**

Run: `PYTHONPATH=. pytest tests/ -v -k "not integration"`
Expected: All tests PASS, no regressions

- [ ] **Step 22: Verify editor CLI help**

Run: `PYTHONPATH=. python -m traffic_detection_kpi.editor_cli --help`
Expected: Shows `--config` and `--video` options

- [ ] **Step 23: Verify main CLI still works**

Run: `PYTHONPATH=. python -m traffic_detection_kpi --help`
Expected: Shows all existing flags (`--config`, `--youtube`, `--rtsp`, `--show`, `--verbose`)
