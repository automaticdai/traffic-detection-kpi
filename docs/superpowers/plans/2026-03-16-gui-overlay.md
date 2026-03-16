# GUI Overlay Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live OpenCV GUI overlay showing detections, lane polygons, and a metrics side panel toggled with `--show`.

**Architecture:** New `FrameAnnotator` in `annotator.py` draws overlays and a side panel on each frame. `MetricsCollector` gains a `snapshot()` method. Pipeline calls the annotator and `cv2.imshow()` when `--show` is active.

**Tech Stack:** OpenCV (drawing + display), numpy (frame concatenation)

**Spec:** `docs/superpowers/specs/2026-03-16-gui-overlay-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `traffic_detection_kpi/annotator.py` | Create | `FrameAnnotator` class — draws lane polygons, bboxes, track labels, metrics side panel |
| `tests/test_annotator.py` | Create | Unit tests for annotator |
| `traffic_detection_kpi/metrics.py` | Modify | Add `snapshot()` method and `_last_lane_assignments` storage |
| `tests/test_metrics.py` | Modify | Add `test_snapshot_returns_current_state` |
| `traffic_detection_kpi/pipeline.py` | Modify | Accept `show` param, call annotator + imshow in frame loop |
| `traffic_detection_kpi/__main__.py` | Modify | Add `--show` CLI flag |

---

## Chunk 1: MetricsCollector.snapshot()

### Task 1: snapshot() — test

**Files:**
- Modify: `tests/test_metrics.py`

- [ ] **Step 1: Write snapshot test**

Append to `tests/test_metrics.py`:

```python
def test_snapshot_returns_current_state():
    mc = MetricsCollector(lane_names=["L1", "L2"], video_fps=30, max_age=20)
    car1 = _make_obj(1, "car", 2)
    bus2 = _make_obj(2, "bus", 5)

    # Run 30 frames so throughput triggers for both
    for _ in range(30):
        mc.update({"L1": [car1], "L2": [bus2]})

    snap = mc.snapshot()

    # Structure
    assert "lanes" in snap
    assert "elapsed_frames" in snap
    assert snap["elapsed_frames"] == 30

    # L1
    l1 = snap["lanes"]["L1"]
    assert l1["queue_length"] == 1
    assert l1["throughput_total"] == 1
    assert l1["throughput_rate"] > 0
    assert l1["avg_dwell"] > 0
    assert l1["vehicle_counts"]["car"] == 1

    # L2
    l2 = snap["lanes"]["L2"]
    assert l2["queue_length"] == 1
    assert l2["throughput_total"] == 1
    assert l2["vehicle_counts"]["bus"] == 1


def test_snapshot_empty_lanes():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    mc.update({"L1": []})

    snap = mc.snapshot()
    l1 = snap["lanes"]["L1"]
    assert l1["queue_length"] == 0
    assert l1["throughput_total"] == 0
    assert l1["avg_dwell"] == 0.0
    assert l1["vehicle_counts"] == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_metrics.py::test_snapshot_returns_current_state -v`
Expected: FAIL — `AttributeError: 'MetricsCollector' object has no attribute 'snapshot'`

### Task 2: snapshot() — implementation

**Files:**
- Modify: `traffic_detection_kpi/metrics.py`

- [ ] **Step 3: Add _last_lane_assignments storage in update()**

In `metrics.py`, at the start of `update()` (after `self.frame_count += 1`), add:

```python
self._last_lane_assignments = lane_assignments
```

- [ ] **Step 4: Add snapshot() method**

Add to `MetricsCollector` after `update()` and before `finalize()`:

```python
def snapshot(self) -> dict:
    """Return current per-lane metrics for live display."""
    duration = self.frame_count / self.video_fps if self.video_fps > 0 else 0.0
    assignments = getattr(self, "_last_lane_assignments", {})

    lanes = {}
    for name in self.lane_names:
        objects = assignments.get(name, [])
        dwell_values = []
        for obj in objects:
            frames = self._dwell_frames.get(obj.track_id, 0)
            dwell_values.append(frames / self.video_fps if self.video_fps > 0 else 0.0)

        total = self._throughput[name]
        lanes[name] = {
            "queue_length": len(objects),
            "throughput_total": total,
            "throughput_rate": total / duration if duration > 0 else 0.0,
            "avg_dwell": sum(dwell_values) / len(dwell_values) if dwell_values else 0.0,
            "vehicle_counts": dict(self._vehicle_counts[name]),
        }

    return {
        "lanes": lanes,
        "elapsed_frames": self.frame_count,
    }
```

- [ ] **Step 5: Run snapshot tests**

Run: `PYTHONPATH=. pytest tests/test_metrics.py -v`
Expected: All 9 tests PASS (7 existing + 2 new)

- [ ] **Step 6: Commit**

```bash
git add traffic_detection_kpi/metrics.py tests/test_metrics.py
git commit -m "feat: add MetricsCollector.snapshot() for live display"
```

---

## Chunk 2: FrameAnnotator

### Task 3: FrameAnnotator — tests

**Files:**
- Create: `tests/test_annotator.py`

- [ ] **Step 7: Write annotator tests**

```python
import numpy as np
import pytest


def _make_snapshot(lane_names):
    """Create a minimal metrics snapshot."""
    return {
        "lanes": {
            name: {
                "queue_length": 0,
                "throughput_total": 0,
                "throughput_rate": 0.0,
                "avg_dwell": 0.0,
                "vehicle_counts": {},
            }
            for name in lane_names
        },
        "elapsed_frames": 0,
    }


class TestFrameAnnotator:
    def test_draw_returns_frame_with_panel(self):
        from traffic_detection_kpi.annotator import FrameAnnotator

        lane_names = ["Lane 1"]
        lane_polygons = [[[100, 100], [200, 100], [200, 300], [100, 300]]]
        annotator = FrameAnnotator(lane_names, lane_polygons, fps=30)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        snapshot = _make_snapshot(lane_names)

        result = annotator.draw(frame, [], {}, snapshot)

        assert result.shape == (480, 940, 3)  # 640 + 300 panel
        assert result.dtype == np.uint8

    def test_draw_with_empty_inputs(self):
        from traffic_detection_kpi.annotator import FrameAnnotator

        lane_names = ["Lane 1", "Lane 2"]
        lane_polygons = [
            [[0, 0], [100, 0], [100, 100]],
            [[200, 0], [300, 0], [300, 100]],
        ]
        annotator = FrameAnnotator(lane_names, lane_polygons, fps=30)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        snapshot = _make_snapshot(lane_names)

        # Should not crash with no objects
        result = annotator.draw(frame, [], {"Lane 1": [], "Lane 2": []}, snapshot)
        assert result.shape == (480, 940, 3)

    def test_draw_with_detections(self):
        from traffic_detection_kpi.annotator import FrameAnnotator
        from traffic_detection_kpi import TrackedObject

        lane_names = ["Lane 1"]
        lane_polygons = [[[0, 0], [640, 0], [640, 480], [0, 480]]]
        annotator = FrameAnnotator(lane_names, lane_polygons, fps=30)

        obj = TrackedObject(
            track_id=1,
            bbox=(100, 100, 200, 200),
            class_id=2,
            class_name="car",
            center=(150, 150),
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        snapshot = _make_snapshot(lane_names)
        snapshot["lanes"]["Lane 1"]["queue_length"] = 1

        result = annotator.draw(frame, [obj], {"Lane 1": [obj]}, snapshot)

        assert result.shape == (480, 940, 3)
        assert result.dtype == np.uint8
        # Verify something was drawn (frame shouldn't be all zeros anymore)
        assert result[:, :640, :].sum() > 0

    def test_lane_colors_are_consistent(self):
        from traffic_detection_kpi.annotator import FrameAnnotator

        lane_names = ["A", "B", "C"]
        lane_polygons = [
            [[0, 0], [10, 0], [10, 10]],
            [[20, 0], [30, 0], [30, 10]],
            [[40, 0], [50, 0], [50, 10]],
        ]
        a1 = FrameAnnotator(lane_names, lane_polygons, fps=30)
        a2 = FrameAnnotator(lane_names, lane_polygons, fps=30)

        assert a1._lane_colors == a2._lane_colors
        assert len(a1._lane_colors) == 3
        # Each lane gets a different color
        assert a1._lane_colors[0] != a1._lane_colors[1]
```

- [ ] **Step 8: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_annotator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'traffic_detection_kpi.annotator'`

### Task 4: FrameAnnotator — implementation

**Files:**
- Create: `traffic_detection_kpi/annotator.py`

- [ ] **Step 9: Implement FrameAnnotator**

```python
from __future__ import annotations

import cv2
import numpy as np

from traffic_detection_kpi import TrackedObject

# BGR color palette for lanes
_PALETTE = [
    (0, 200, 0),      # green
    (200, 100, 0),     # blue
    (0, 140, 255),     # orange
    (0, 0, 200),       # red
    (200, 200, 0),     # cyan
    (200, 0, 200),     # magenta
    (0, 200, 200),     # yellow
    (200, 200, 200),   # light gray
]

_PANEL_WIDTH = 300
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GRAY = (128, 128, 128)


class FrameAnnotator:
    def __init__(
        self,
        lane_names: list[str],
        lane_polygons: list[list[list[int]]],
        fps: int,
    ) -> None:
        self._lane_names = lane_names
        self._lane_polygons = [np.array(p, dtype=np.int32) for p in lane_polygons]
        self._fps = fps
        self._lane_colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(lane_names))]
        self._lane_color_map = dict(zip(lane_names, self._lane_colors))

    def draw(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        lane_assignments: dict[str, list[TrackedObject]],
        metrics_snapshot: dict,
    ) -> np.ndarray:
        canvas = frame.copy()
        self._draw_lanes(canvas)
        self._draw_boxes(canvas, tracked_objects, lane_assignments)
        panel = self._draw_panel(frame.shape[0], metrics_snapshot)
        return np.hstack([canvas, panel])

    def _draw_lanes(self, canvas: np.ndarray) -> None:
        overlay = canvas.copy()
        for poly, color in zip(self._lane_polygons, self._lane_colors):
            cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        for poly, color in zip(self._lane_polygons, self._lane_colors):
            cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=2)

    def _draw_boxes(
        self,
        canvas: np.ndarray,
        tracked_objects: list[TrackedObject],
        lane_assignments: dict[str, list[TrackedObject]],
    ) -> None:
        # Build track_id -> lane color map
        id_to_color: dict[int, tuple[int, int, int]] = {}
        for lane_name, objects in lane_assignments.items():
            color = self._lane_color_map.get(lane_name, _GRAY)
            for obj in objects:
                id_to_color[obj.track_id] = color

        for obj in tracked_objects:
            color = id_to_color.get(obj.track_id, _GRAY)
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"#{obj.track_id} {obj.class_name}"
            cv2.putText(canvas, label, (x1, y1 - 8), _FONT, 0.5, color, 1, cv2.LINE_AA)

    def _draw_panel(self, height: int, snapshot: dict) -> np.ndarray:
        panel = np.zeros((height, _PANEL_WIDTH, 3), dtype=np.uint8)
        y = 30
        cv2.putText(panel, "Live Metrics", (10, y), _FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 35

        lanes_data = snapshot.get("lanes", {})
        for lane_name, color in zip(self._lane_names, self._lane_colors):
            data = lanes_data.get(lane_name, {})
            cv2.putText(panel, lane_name, (10, y), _FONT, 0.55, color, 1, cv2.LINE_AA)
            y += 22
            cv2.putText(panel, f"  Vehicles: {data.get('queue_length', 0)}", (10, y), _FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            y += 18
            total = data.get("throughput_total", 0)
            rate = data.get("throughput_rate", 0.0)
            cv2.putText(panel, f"  Throughput: {total} ({rate:.1f}/s)", (10, y), _FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            y += 18
            dwell = data.get("avg_dwell", 0.0)
            cv2.putText(panel, f"  Avg dwell: {dwell:.1f}s", (10, y), _FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            y += 18
            counts = data.get("vehicle_counts", {})
            if counts:
                parts = "  " + "  ".join(f"{k}: {v}" for k, v in counts.items())
                cv2.putText(panel, parts, (10, y), _FONT, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
            y += 25

        # Footer
        elapsed_frames = snapshot.get("elapsed_frames", 0)
        elapsed_sec = elapsed_frames / self._fps if self._fps > 0 else 0.0
        footer = f"FPS: {self._fps} | Elapsed: {elapsed_sec:.0f}s"
        cv2.putText(panel, footer, (10, height - 15), _FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        return panel
```

- [ ] **Step 10: Run annotator tests**

Run: `PYTHONPATH=. pytest tests/test_annotator.py -v`
Expected: All 4 tests PASS

- [ ] **Step 11: Commit**

```bash
git add traffic_detection_kpi/annotator.py tests/test_annotator.py
git commit -m "feat: add FrameAnnotator with lane overlays, bbox drawing, and metrics panel"
```

---

## Chunk 3: Pipeline and CLI Integration

### Task 5: Pipeline integration

**Files:**
- Modify: `traffic_detection_kpi/pipeline.py`

- [ ] **Step 12: Add show parameter and annotator to pipeline**

In `pipeline.py`:

Change `__init__` signature to accept `show`:

```python
def __init__(self, config: Config, source: VideoSource | None = None, show: bool = False):
    self.config = config
    self.source = source
    self.show = show
```

(Keep the rest of `__init__` unchanged.)

Add import at top:

```python
from traffic_detection_kpi.annotator import FrameAnnotator
```

In `run()`, after `source` is resolved and before the `try` block, add annotator construction:

```python
annotator = None
if self.show:
    annotator = FrameAnnotator(
        lane_names=[l.name for l in self.lanes],
        lane_polygons=[l.polygon for l in self.config.lanes],
        fps=source.fps,
    )
```

Change the `_run_loop` call to pass `annotator`:

```python
self._run_loop(source, shutdown_check=lambda: shutdown, annotator=annotator)
```

In the `finally` block, after `source.release()`:

```python
if annotator:
    cv2.destroyAllWindows()
```

- [ ] **Step 13: Add annotator call in _run_loop**

Change `_run_loop` signature:

```python
def _run_loop(self, source: VideoSource, shutdown_check, annotator=None):
```

After `metrics.update(assignments)` (line 72 in current pipeline.py), add:

```python
if annotator:
    try:
        snap = metrics.snapshot()
        annotated = annotator.draw(frame, tracked, assignments, snap)
        cv2.imshow("Traffic Detection KPI", annotated)
        cv2.waitKey(1)
    except cv2.error:
        logger.warning("No display available, disabling GUI overlay")
        annotator = None
```

- [ ] **Step 14: Run all tests**

Run: `PYTHONPATH=. pytest tests/ -v -k "not integration"`
Expected: All tests PASS (no regressions — existing pipeline tests use `show=False` by default)

- [ ] **Step 15: Commit**

```bash
git add traffic_detection_kpi/pipeline.py
git commit -m "feat: integrate FrameAnnotator into pipeline with --show support"
```

### Task 6: CLI --show flag

**Files:**
- Modify: `traffic_detection_kpi/__main__.py`

- [ ] **Step 16: Add --show flag**

In `__main__.py`, add after the `--rtsp` argument:

```python
parser.add_argument(
    "--show", action="store_true", help="Show live GUI overlay"
)
```

Change the pipeline construction line:

```python
pipeline = VideoPipeline(config, source=source, show=args.show)
```

- [ ] **Step 17: Run all tests**

Run: `PYTHONPATH=. pytest tests/ -v -k "not integration"`
Expected: All tests PASS

- [ ] **Step 18: Verify CLI help**

Run: `PYTHONPATH=. python -m traffic_detection_kpi --help`
Expected: Shows `--show` in the options list

- [ ] **Step 19: Commit**

```bash
git add traffic_detection_kpi/__main__.py
git commit -m "feat: add --show CLI flag for live GUI overlay"
```

---

## Chunk 4: Final Verification

### Task 7: Full regression check

- [ ] **Step 20: Run complete test suite**

Run: `PYTHONPATH=. pytest tests/ -v -k "not integration"`
Expected: All tests PASS, no regressions

- [ ] **Step 21: Verify CLI help includes all flags**

Run: `PYTHONPATH=. python -m traffic_detection_kpi --help`
Expected: Shows `--config`, `--youtube`, `--rtsp`, `--show`, `--verbose`
