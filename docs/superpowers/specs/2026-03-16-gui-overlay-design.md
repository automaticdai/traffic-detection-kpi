# GUI Overlay Design

**Date:** 2026-03-16
**Status:** Approved
**Scope:** Add a live OpenCV GUI overlay showing detections, lane regions, and a metrics side panel

## Problem

The pipeline processes frames silently with no visual feedback. Users need to verify that detections, tracking, and lane assignments are working correctly, and to monitor live metrics during processing.

## Requirements

- Display each frame with overlaid lane polygons, bounding boxes, track IDs, and class labels
- Show a side panel with per-lane metrics: vehicle count, throughput, avg dwell time, class breakdown
- Toggle via `--show` CLI flag (off by default)
- Must not affect pipeline output (JSON, charts) or performance significantly

## Non-Requirements

- Recording annotated video to file
- Interactive controls (pause, step, zoom)
- Configurable colors or layout

## Design

### FrameAnnotator Module

A new module `traffic_detection_kpi/annotator.py` with a `FrameAnnotator` class.

**Interface:**

```python
class FrameAnnotator:
    def __init__(self, lane_names: list[str], lane_polygons: list[list[list[int]]], fps: int) -> None:
        """Initialize with lane names, their polygon coordinates, and source FPS."""
        ...

    def draw(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        lane_assignments: dict[str, list[TrackedObject]],
        metrics_snapshot: dict,
    ) -> np.ndarray:
        """Return a new frame with overlays and side panel appended."""
        ...
```

**Overlay elements drawn on the video frame:**
- Lane polygons: semi-transparent colored fill + solid border, one consistent color per lane (assigned by index)
- Bounding boxes: colored rectangle around each tracked object matching its lane color, with track ID and class label text above the box
- Objects not in any lane get a neutral gray box

**Side panel:**
- Fixed-width dark panel (300px) appended to the right of the frame via numpy concatenation
- Layout (top to bottom):
  - Title: "Live Metrics" in white
  - Per lane section (repeated for each lane):
    - Lane name in the lane's assigned color
    - `Vehicles: N` (current queue length)
    - `Throughput: N (R/s)` (total count and rate)
    - `Avg dwell: X.Xs`
    - Class breakdown: `car: N  bus: N  ...`
  - Footer: `FPS: N | Elapsed: Xs`
- Text rendered with `cv2.putText` using `FONT_HERSHEY_SIMPLEX`

**Color palette:**
A fixed list of distinct colors assigned to lanes by index (e.g., green, blue, orange, red, cyan, magenta). Wraps around if more lanes than colors.

### MetricsCollector Changes

Add a `snapshot()` method to `MetricsCollector` that returns the current state without modifying anything:

```python
def snapshot(self) -> dict:
    """Return current per-lane metrics for display.

    Returns:
        {
            "lanes": {
                "Lane 1": {
                    "queue_length": int,
                    "throughput_total": int,
                    "throughput_rate": float,
                    "avg_dwell": float,
                    "vehicle_counts": {"car": int, ...},
                },
                ...
            },
            "elapsed_frames": int,
        }
    """
```

- `queue_length`: number of tracked objects currently in the lane (from the most recent `update()` call)
- `throughput_total`: cumulative count of vehicles that passed the 1-second threshold
- `throughput_rate`: `throughput_total / (elapsed_frames / fps)` if elapsed > 0, else 0
- `avg_dwell`: average dwell time in seconds across currently tracked objects in the lane
- `vehicle_counts`: class breakdown dict

To compute `queue_length` and `avg_dwell` per lane, `update()` must store the most recent `lane_assignments` dict as `self._last_lane_assignments`. This is the one new piece of state needed — `snapshot()` reads it to determine which tracks are currently in each lane and their dwell times. The `elapsed_frames` field in the snapshot is sourced from the existing `self.frame_count` attribute.

### Pipeline Changes

**`pipeline.py`:**
- `VideoPipeline.__init__` accepts `show: bool = False` and stores it as `self.show`
- In `run()`, after `source` is resolved (either injected or created as `FileSource`), if `self.show` is True, construct `FrameAnnotator(lane_names, lane_polygons, source.fps)` and pass it to `_run_loop`
- In `_run_loop`, after `metrics.update()`:
  - If annotator is provided: call `metrics.snapshot()`, pass to `annotator.draw()`, display with `cv2.imshow("Traffic Detection KPI", annotated)` and `cv2.waitKey(1)`
- After the loop, if annotator was used: call `cv2.destroyAllWindows()`
- **Headless environments:** If `cv2.imshow()` raises an error (no display available), log a warning and disable the overlay for the rest of the run rather than crashing the pipeline

**`__main__.py`:**
- Add `--show` flag: `parser.add_argument("--show", action="store_true", help="Show live GUI overlay")`
- Pass to pipeline: `VideoPipeline(config, source=source, show=args.show)`

### Files Changed

| File | Change |
|------|--------|
| `traffic_detection_kpi/annotator.py` | **New** — FrameAnnotator class |
| `traffic_detection_kpi/metrics.py` | Add `snapshot()` method |
| `traffic_detection_kpi/pipeline.py` | Accept `show` param, call annotator + imshow |
| `traffic_detection_kpi/__main__.py` | Add `--show` CLI flag |
| `tests/test_annotator.py` | **New** — annotator unit tests |
| `tests/test_metrics.py` | Add snapshot() test |

### Testing

**New tests in `tests/test_annotator.py`:**
- `test_draw_returns_frame_with_panel`: using a 640x480 input frame, verify output shape is (480, 940, 3) — input width + 300px panel, same height
- `test_draw_with_empty_inputs`: no tracked objects, no lane assignments — should not crash, still returns frame with panel
- `test_draw_with_detections`: verify output frame is valid numpy array with expected dimensions
- `test_lane_colors_are_consistent`: same lane always gets the same color across calls

**New test in `tests/test_metrics.py`:**
- `test_snapshot_returns_current_state`: after a few update() calls, verify snapshot() returns correct structure with expected values

**Existing tests:** No changes required.
