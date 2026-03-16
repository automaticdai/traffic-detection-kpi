# Traffic Detection KPI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular Python package that processes recorded traffic video to compute per-lane KPIs (throughput, queue length, dwell time, vehicle class breakdown) and outputs JSON + matplotlib charts.

**Architecture:** Modular Python package with 7 modules following a linear pipeline: config loading → YOLO detection → DeepSort tracking → lane classification → metrics computation → report generation. Each module has a single responsibility and communicates through well-defined dataclasses.

**Tech Stack:** Python 3.10+, ultralytics (YOLO11m), deep-sort-realtime, OpenCV, Shapely, matplotlib/seaborn, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-traffic-detection-kpi-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies |
| `traffic_detection_kpi/__init__.py` | Package init, exports dataclasses |
| `traffic_detection_kpi/__main__.py` | CLI entry point |
| `traffic_detection_kpi/config.py` | YAML loading, validation, Config dataclass |
| `traffic_detection_kpi/detection.py` | YoloDetector wrapping YOLO model |
| `traffic_detection_kpi/tracking.py` | DeepSortTracker wrapping DeepSort |
| `traffic_detection_kpi/lanes.py` | LaneZone polygon + classification |
| `traffic_detection_kpi/metrics.py` | MetricsCollector + MetricsResult |
| `traffic_detection_kpi/pipeline.py` | VideoPipeline orchestration |
| `traffic_detection_kpi/reporting.py` | JSON + chart generation |
| `configs/example.yaml` | Example config |
| `tests/test_config.py` | Config unit tests |
| `tests/test_lanes.py` | Lane classification tests |
| `tests/test_metrics.py` | Metrics computation tests |
| `tests/test_detection.py` | Detection wrapper tests |
| `tests/test_tracking.py` | Tracking wrapper tests |
| `tests/test_reporting.py` | Report generation tests |
| `tests/conftest.py` | Pytest config (integration marker) |
| `tests/test_pipeline.py` | Integration test |

---

## Chunk 1: Project Setup & Data Types

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `traffic_detection_kpi/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "traffic-detection-kpi"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "ultralytics>=8.0",
    "deep-sort-realtime>=1.3",
    "opencv-python>=4.8",
    "shapely>=2.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[project.scripts]
traffic-kpi = "traffic_detection_kpi.__main__:main"
```

- [ ] **Step 2: Create `traffic_detection_kpi/__init__.py` with shared dataclasses**

```python
from dataclasses import dataclass, field


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x1, y1, w, h (xywh)
    class_id: int
    class_name: str
    confidence: float


@dataclass
class TrackedObject:
    track_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (ltrb)
    class_id: int
    class_name: str
    center: tuple[int, int]  # midpoint of ltrb bbox


@dataclass
class LaneMetrics:
    throughput_total: int = 0
    throughput_rate_avg: float = 0.0
    vehicle_counts: dict[str, int] = field(default_factory=dict)
    queue_length_timeseries: list[int] = field(default_factory=list)
    avg_dwell_time_timeseries: list[float] = field(default_factory=list)


@dataclass
class MetricsResult:
    video_path: str
    total_frames: int
    duration_seconds: float
    fps: int
    lanes: dict[str, LaneMetrics] = field(default_factory=dict)
```

- [ ] **Step 3: Verify import works**

Run: `python -c "from traffic_detection_kpi import Detection, TrackedObject, LaneMetrics, MetricsResult; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml traffic_detection_kpi/__init__.py
git commit -m "feat: project scaffolding with shared dataclasses"
```

---

### Task 2: Config module

**Files:**
- Create: `traffic_detection_kpi/config.py`
- Create: `tests/test_config.py`
- Create: `configs/example.yaml`

- [ ] **Step 1: Write failing tests for config loading**

Create `tests/test_config.py`:

```python
import pytest
import tempfile
import os
from pathlib import Path


def _write_yaml(tmp_path: Path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return str(p)


VALID_YAML = """\
video_path: "test.mp4"
output_dir: "./output"
model:
  path: "yolo11m.pt"
  confidence: 0.2
  classes: [car, motorcycle, bus, truck]
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


def test_load_valid_config(tmp_path):
    from traffic_detection_kpi.config import load_config
    path = _write_yaml(tmp_path, VALID_YAML)
    config = load_config(path)
    assert config.video_path == "test.mp4"
    assert config.model.confidence == 0.2
    assert len(config.lanes) == 1
    assert config.lanes[0].name == "Lane 1"


def test_class_name_to_coco_id(tmp_path):
    from traffic_detection_kpi.config import load_config
    path = _write_yaml(tmp_path, VALID_YAML)
    config = load_config(path)
    assert config.model.class_ids == [2, 3, 5, 7]


def test_missing_video_path(tmp_path):
    from traffic_detection_kpi.config import load_config
    yaml = VALID_YAML.replace('video_path: "test.mp4"\n', '')
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="video_path"):
        load_config(path)


def test_invalid_class_name(tmp_path):
    from traffic_detection_kpi.config import load_config
    yaml = VALID_YAML.replace("classes: [car, motorcycle, bus, truck]",
                               "classes: [car, airplane]")
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="airplane"):
        load_config(path)


def test_polygon_needs_at_least_3_points(tmp_path):
    from traffic_detection_kpi.config import load_config
    yaml = VALID_YAML.replace(
        "polygon: [[0, 0], [100, 0], [100, 100], [0, 100]]",
        "polygon: [[0, 0], [100, 0]]"
    )
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="polygon"):
        load_config(path)


def test_self_intersecting_polygon_rejected(tmp_path):
    from traffic_detection_kpi.config import load_config
    # Bowtie shape — self-intersecting
    yaml = VALID_YAML.replace(
        "polygon: [[0, 0], [100, 0], [100, 100], [0, 100]]",
        "polygon: [[0, 0], [100, 100], [100, 0], [0, 100]]"
    )
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="self-intersecting"):
        load_config(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'traffic_detection_kpi.config'`

- [ ] **Step 3: Implement `config.py`**

Create `traffic_detection_kpi/config.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from shapely.geometry import Polygon as ShapelyPolygon


COCO_CLASS_MAP = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


@dataclass
class LaneConfig:
    name: str
    polygon: list[list[int]]


@dataclass
class ModelConfig:
    path: str
    confidence: float
    classes: list[str]
    class_ids: list[int] = field(default_factory=list)


@dataclass
class TrackerConfig:
    type: str
    max_age: int
    n_init: int
    max_cosine_distance: float
    embedder: str


@dataclass
class Config:
    video_path: str
    output_dir: str
    model: ModelConfig
    tracker: TrackerConfig
    lanes: list[LaneConfig]


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    _validate_required(raw, ["video_path", "output_dir", "model", "tracker", "lanes"])

    model_raw = raw["model"]
    _validate_required(model_raw, ["path", "confidence", "classes"])
    for cls in model_raw["classes"]:
        if cls not in COCO_CLASS_MAP:
            raise ValueError(f"Unknown class name: '{cls}'. Valid: {list(COCO_CLASS_MAP.keys())}")
    class_ids = [COCO_CLASS_MAP[c] for c in model_raw["classes"]]
    model = ModelConfig(
        path=model_raw["path"],
        confidence=model_raw["confidence"],
        classes=model_raw["classes"],
        class_ids=class_ids,
    )

    tracker_raw = raw["tracker"]
    _validate_required(tracker_raw, ["type", "max_age", "n_init", "max_cosine_distance", "embedder"])
    tracker = TrackerConfig(**{k: tracker_raw[k] for k in ["type", "max_age", "n_init", "max_cosine_distance", "embedder"]})

    lanes = []
    for i, lane_raw in enumerate(raw["lanes"]):
        _validate_required(lane_raw, ["name", "polygon"])
        polygon = lane_raw["polygon"]
        if len(polygon) < 3:
            raise ValueError(f"Lane '{lane_raw['name']}' polygon must have at least 3 points, got {len(polygon)}")
        if not ShapelyPolygon(polygon).is_valid:
            raise ValueError(f"Lane '{lane_raw['name']}' polygon is self-intersecting")
        lanes.append(LaneConfig(name=lane_raw["name"], polygon=polygon))

    return Config(
        video_path=raw["video_path"],
        output_dir=raw["output_dir"],
        model=model,
        tracker=tracker,
        lanes=lanes,
    )


def _validate_required(data: dict, keys: list[str]):
    for key in keys:
        if key not in data:
            raise ValueError(f"Missing required field: '{key}'")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Create example config**

Create `configs/example.yaml`:

```yaml
video_path: "../trafficData/four lanes.mp4"
output_dir: "./output"
model:
  path: "yolo11m.pt"
  confidence: 0.2
  classes: [car, motorcycle, bus, truck]
tracker:
  type: deepsort
  max_age: 20
  n_init: 2
  max_cosine_distance: 0.8
  embedder: mobilenet
lanes:
  - name: "Lane 1"
    polygon: [[300, 570], [750, 570], [650, 150], [500, 150]]
  - name: "Lane 2"
    polygon: [[610, 550], [750, 550], [650, 150], [596, 150]]
  - name: "Lane 3"
    polygon: [[770, 550], [900, 550], [770, 200], [670, 200]]
```

- [ ] **Step 6: Commit**

```bash
git add traffic_detection_kpi/config.py tests/test_config.py configs/example.yaml
git commit -m "feat: config module with YAML loading and validation"
```

---

### Task 3: Lanes module

**Files:**
- Create: `traffic_detection_kpi/lanes.py`
- Create: `tests/test_lanes.py`

- [ ] **Step 1: Write failing tests for lane classification**

Create `tests/test_lanes.py`:

```python
from traffic_detection_kpi import TrackedObject
from traffic_detection_kpi.lanes import LaneZone


def _make_obj(track_id: int, center: tuple[int, int]) -> TrackedObject:
    cx, cy = center
    return TrackedObject(
        track_id=track_id,
        bbox=(cx - 10, cy - 10, cx + 10, cy + 10),
        class_id=2,
        class_name="car",
        center=center,
    )


def test_contains_point_inside():
    lane = LaneZone("L1", [[0, 0], [100, 0], [100, 100], [0, 100]])
    assert lane.contains((50, 50)) is True


def test_contains_point_outside():
    lane = LaneZone("L1", [[0, 0], [100, 0], [100, 100], [0, 100]])
    assert lane.contains((200, 200)) is False


def test_classify_assigns_to_correct_lane():
    lane1 = LaneZone("Left", [[0, 0], [50, 0], [50, 100], [0, 100]])
    lane2 = LaneZone("Right", [[50, 0], [100, 0], [100, 100], [50, 100]])
    obj_left = _make_obj(1, (25, 50))
    obj_right = _make_obj(2, (75, 50))
    result = LaneZone.classify([lane1, lane2], [obj_left, obj_right])
    assert len(result["Left"]) == 1
    assert result["Left"][0].track_id == 1
    assert len(result["Right"]) == 1
    assert result["Right"][0].track_id == 2


def test_classify_first_match_wins():
    # Overlapping lanes — object at (50, 50) is in both
    lane1 = LaneZone("First", [[0, 0], [100, 0], [100, 100], [0, 100]])
    lane2 = LaneZone("Second", [[25, 25], [75, 25], [75, 75], [25, 75]])
    obj = _make_obj(1, (50, 50))
    result = LaneZone.classify([lane1, lane2], [obj])
    assert len(result["First"]) == 1
    assert len(result["Second"]) == 0


def test_classify_excludes_objects_outside_all_lanes():
    lane = LaneZone("L1", [[0, 0], [50, 0], [50, 50], [0, 50]])
    obj = _make_obj(1, (200, 200))
    result = LaneZone.classify([lane], [obj])
    assert len(result["L1"]) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lanes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'traffic_detection_kpi.lanes'`

- [ ] **Step 3: Implement `lanes.py`**

Create `traffic_detection_kpi/lanes.py`:

```python
from shapely.geometry import Point, Polygon

from traffic_detection_kpi import TrackedObject


class LaneZone:
    def __init__(self, name: str, polygon_coords: list[list[int]]):
        self.name = name
        self.polygon = Polygon(polygon_coords)

    def contains(self, point: tuple[int, int]) -> bool:
        return self.polygon.contains(Point(point))

    @classmethod
    def classify(
        cls, lanes: list["LaneZone"], tracked_objects: list[TrackedObject]
    ) -> dict[str, list[TrackedObject]]:
        result: dict[str, list[TrackedObject]] = {lane.name: [] for lane in lanes}
        for obj in tracked_objects:
            for lane in lanes:
                if lane.contains(obj.center):
                    result[lane.name].append(obj)
                    break  # first-match wins
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_lanes.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add traffic_detection_kpi/lanes.py tests/test_lanes.py
git commit -m "feat: lane zone classification with first-match-wins"
```

---

### Task 4: Metrics module

**Files:**
- Create: `traffic_detection_kpi/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests for metrics computation**

Create `tests/test_metrics.py`:

```python
from traffic_detection_kpi import TrackedObject, LaneMetrics
from traffic_detection_kpi.metrics import MetricsCollector


def _make_obj(track_id: int, class_name: str = "car", class_id: int = 2) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=(0, 0, 20, 20),
        class_id=class_id,
        class_name=class_name,
        center=(10, 10),
    )


def test_queue_length_counts_objects_per_frame():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    mc.update({"L1": [_make_obj(1), _make_obj(2)]})
    # Internal current queue should be 2
    # After 30 frames (1 second), timeseries should sample
    for _ in range(29):
        mc.update({"L1": [_make_obj(1), _make_obj(2)]})
    result = mc.finalize()
    assert result.lanes["L1"].queue_length_timeseries[0] == 2


def test_throughput_after_one_second():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    # Same object present for 30 frames = 1 second → counts as throughput
    for _ in range(30):
        mc.update({"L1": [_make_obj(1)]})
    result = mc.finalize()
    assert result.lanes["L1"].throughput_total == 1


def test_throughput_not_counted_before_threshold():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    # Object present for only 10 frames — should not count
    for _ in range(10):
        mc.update({"L1": [_make_obj(1)]})
    result = mc.finalize()
    assert result.lanes["L1"].throughput_total == 0


def test_vehicle_class_breakdown():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    car = _make_obj(1, "car", 2)
    truck = _make_obj(2, "truck", 7)
    for _ in range(30):
        mc.update({"L1": [car, truck]})
    result = mc.finalize()
    assert result.lanes["L1"].vehicle_counts["car"] == 1
    assert result.lanes["L1"].vehicle_counts["truck"] == 1


def test_dwell_time_timeseries():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    obj = _make_obj(1)
    for _ in range(30):
        mc.update({"L1": [obj]})
    result = mc.finalize()
    # After 30 frames at 30fps, avg dwell = ~0.5s (average of 1/30..30/30)
    assert len(result.lanes["L1"].avg_dwell_time_timeseries) == 1
    assert result.lanes["L1"].avg_dwell_time_timeseries[0] > 0


def test_track_pruning():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=5)
    obj = _make_obj(1)
    # Object appears for 3 frames then disappears
    for _ in range(3):
        mc.update({"L1": [obj]})
    # Object absent for max_age frames → should be pruned
    for _ in range(6):
        mc.update({"L1": []})
    # Check internal state: track_id 1 should be pruned
    assert 1 not in mc._dwell_frames


def test_multiple_lanes():
    mc = MetricsCollector(lane_names=["L1", "L2"], video_fps=30, max_age=20)
    obj1 = _make_obj(1)
    obj2 = _make_obj(2)
    for _ in range(30):
        mc.update({"L1": [obj1], "L2": [obj2]})
    result = mc.finalize()
    assert result.lanes["L1"].throughput_total == 1
    assert result.lanes["L2"].throughput_total == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'traffic_detection_kpi.metrics'`

- [ ] **Step 3: Implement `metrics.py`**

Create `traffic_detection_kpi/metrics.py`:

```python
from collections import defaultdict

from traffic_detection_kpi import TrackedObject, LaneMetrics, MetricsResult


class MetricsCollector:
    def __init__(self, lane_names: list[str], video_fps: int, max_age: int = 20):
        self.lane_names = lane_names
        self.video_fps = video_fps
        self.max_age = max_age
        self.frame_count = 0

        # Per-track state
        self._dwell_frames: dict[int, int] = {}  # track_id -> frames seen
        self._last_seen: dict[int, int] = {}  # track_id -> last frame number
        self._track_lane: dict[int, str] = {}  # track_id -> lane_name
        self._track_class: dict[int, str] = {}  # track_id -> class_name

        # Per-lane accumulators
        self._throughput: dict[str, int] = {name: 0 for name in lane_names}
        self._counted_ids: dict[str, set[int]] = {name: set() for name in lane_names}
        self._vehicle_counts: dict[str, dict[str, int]] = {name: defaultdict(int) for name in lane_names}

        # Time-series (sampled every video_fps frames = 1 second)
        self._queue_ts: dict[str, list[int]] = {name: [] for name in lane_names}
        self._dwell_ts: dict[str, list[float]] = {name: [] for name in lane_names}

    def update(self, lane_assignments: dict[str, list[TrackedObject]]):
        self.frame_count += 1

        # Track which IDs are seen this frame
        seen_this_frame: set[int] = set()

        # Per-lane queue counts for this frame
        lane_queue: dict[str, int] = {name: 0 for name in self.lane_names}
        lane_dwell_values: dict[str, list[float]] = {name: [] for name in self.lane_names}

        for lane_name, objects in lane_assignments.items():
            lane_queue[lane_name] = len(objects)
            for obj in objects:
                seen_this_frame.add(obj.track_id)
                self._last_seen[obj.track_id] = self.frame_count
                self._track_lane[obj.track_id] = lane_name
                self._track_class[obj.track_id] = obj.class_name

                # Update dwell frame count
                if obj.track_id not in self._dwell_frames:
                    self._dwell_frames[obj.track_id] = 0
                self._dwell_frames[obj.track_id] += 1

                # Check throughput threshold
                if (self._dwell_frames[obj.track_id] == self.video_fps
                        and obj.track_id not in self._counted_ids[lane_name]):
                    self._throughput[lane_name] += 1
                    self._counted_ids[lane_name].add(obj.track_id)
                    self._vehicle_counts[lane_name][obj.class_name] += 1

                # Collect dwell time for averaging
                dwell_seconds = self._dwell_frames[obj.track_id] / self.video_fps
                lane_dwell_values[lane_name].append(dwell_seconds)

        # Prune stale tracks
        stale_ids = [
            tid for tid, last in self._last_seen.items()
            if self.frame_count - last > self.max_age
        ]
        for tid in stale_ids:
            self._dwell_frames.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._track_lane.pop(tid, None)
            self._track_class.pop(tid, None)

        # Sample time-series every second
        if self.frame_count % self.video_fps == 0:
            for lane_name in self.lane_names:
                self._queue_ts[lane_name].append(lane_queue[lane_name])
                dwell_vals = lane_dwell_values[lane_name]
                avg_dwell = sum(dwell_vals) / len(dwell_vals) if dwell_vals else 0.0
                self._dwell_ts[lane_name].append(avg_dwell)

    def finalize(self, video_path: str = "", total_frames: int = 0) -> MetricsResult:
        duration = self.frame_count / self.video_fps if self.video_fps > 0 else 0.0
        lanes: dict[str, LaneMetrics] = {}
        for name in self.lane_names:
            total = self._throughput[name]
            lanes[name] = LaneMetrics(
                throughput_total=total,
                throughput_rate_avg=total / duration if duration > 0 else 0.0,
                vehicle_counts=dict(self._vehicle_counts[name]),
                queue_length_timeseries=list(self._queue_ts[name]),
                avg_dwell_time_timeseries=list(self._dwell_ts[name]),
            )
        return MetricsResult(
            video_path=video_path,
            total_frames=total_frames or self.frame_count,
            duration_seconds=duration,
            fps=self.video_fps,
            lanes=lanes,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add traffic_detection_kpi/metrics.py tests/test_metrics.py
git commit -m "feat: metrics collector with throughput, dwell time, queue length"
```

---

## Chunk 2: Detection, Tracking & Reporting

### Task 5: Detection module

**Files:**
- Create: `traffic_detection_kpi/detection.py`
- Create: `tests/test_detection.py`

- [ ] **Step 1: Write failing tests for detection**

Create `tests/test_detection.py`:

```python
from unittest.mock import MagicMock, patch
from traffic_detection_kpi import Detection


def test_detect_filters_by_allowed_classes():
    from traffic_detection_kpi.detection import YoloDetector

    # Mock YOLO model and results
    mock_box1 = MagicMock()
    mock_box1.xyxy = [MagicMock(__iter__=lambda s: iter([100, 200, 150, 250]))]
    mock_box1.xyxy[0].__iter__ = lambda s: iter([100, 200, 150, 250])
    mock_box1.cls = [MagicMock(item=lambda: 2)]  # car
    mock_box1.conf = [MagicMock(item=lambda: 0.9)]

    mock_box2 = MagicMock()
    mock_box2.xyxy = [MagicMock()]
    mock_box2.xyxy[0].__iter__ = lambda s: iter([100, 200, 150, 250])
    mock_box2.cls = [MagicMock(item=lambda: 0)]  # person — filtered out
    mock_box2.conf = [MagicMock(item=lambda: 0.8)]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box1, mock_box2]
    mock_result.names = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    with patch("traffic_detection_kpi.detection.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        MockYOLO.return_value = mock_model

        detector = YoloDetector(
            model_path="fake.pt",
            confidence=0.2,
            class_filter=["car", "truck"],
        )
        frame = MagicMock()
        detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].class_name == "car"


def test_detect_returns_detection_dataclass():
    from traffic_detection_kpi.detection import YoloDetector

    mock_box = MagicMock()
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].__iter__ = lambda s: iter([100, 200, 160, 270])
    mock_box.cls = [MagicMock(item=lambda: 2)]
    mock_box.conf = [MagicMock(item=lambda: 0.85)]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {2: "car"}

    with patch("traffic_detection_kpi.detection.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        MockYOLO.return_value = mock_model

        detector = YoloDetector("fake.pt", 0.2, ["car"])
        detections = detector.detect(MagicMock())

    assert isinstance(detections[0], Detection)
    assert detections[0].bbox == (100, 200, 60, 70)  # converted to xywh
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_detection.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `detection.py`**

Create `traffic_detection_kpi/detection.py`:

```python
from ultralytics import YOLO

from traffic_detection_kpi import Detection


class YoloDetector:
    def __init__(self, model_path: str, confidence: float, class_filter: list[str]):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.allowed_names = set(class_filter)

    def detect(self, frame) -> list[Detection]:
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        result = results[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = result.names.get(class_id, "")
            if class_name not in self.allowed_names:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0].item())
            detections.append(Detection(
                bbox=(x1, y1, w, h),
                class_id=class_id,
                class_name=class_name,
                confidence=conf,
            ))
        return detections
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_detection.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add traffic_detection_kpi/detection.py tests/test_detection.py
git commit -m "feat: YOLO detection wrapper with class filtering"
```

---

### Task 6: Tracking module

**Files:**
- Create: `traffic_detection_kpi/tracking.py`
- Create: `tests/test_tracking.py`

- [ ] **Step 1: Write failing tests for tracking**

Create `tests/test_tracking.py`:

```python
from unittest.mock import MagicMock, patch
from traffic_detection_kpi import Detection, TrackedObject


def test_track_returns_tracked_objects():
    from traffic_detection_kpi.tracking import DeepSortTracker
    from traffic_detection_kpi.config import TrackerConfig

    config = TrackerConfig(
        type="deepsort", max_age=20, n_init=2,
        max_cosine_distance=0.8, embedder="mobilenet",
    )

    mock_track = MagicMock()
    mock_track.is_confirmed.return_value = True
    mock_track.track_id = 42
    mock_track.to_ltrb.return_value = [100, 200, 300, 400]
    mock_track.get_det_class.return_value = "car"

    with patch("traffic_detection_kpi.tracking.DeepSort") as MockDS:
        mock_ds = MagicMock()
        mock_ds.update_tracks.return_value = [mock_track]
        MockDS.return_value = mock_ds

        tracker = DeepSortTracker(config)
        detections = [Detection(bbox=(100, 200, 200, 200), class_id=2, class_name="car", confidence=0.9)]
        result = tracker.track(detections, MagicMock())

    assert len(result) == 1
    assert isinstance(result[0], TrackedObject)
    assert result[0].track_id == 42
    assert result[0].bbox == (100, 200, 300, 400)
    assert result[0].center == (200, 300)  # midpoint of ltrb
    assert result[0].class_name == "car"
    assert result[0].class_id == 2


def test_track_skips_unconfirmed():
    from traffic_detection_kpi.tracking import DeepSortTracker
    from traffic_detection_kpi.config import TrackerConfig

    config = TrackerConfig(
        type="deepsort", max_age=20, n_init=2,
        max_cosine_distance=0.8, embedder="mobilenet",
    )

    mock_track = MagicMock()
    mock_track.is_confirmed.return_value = False

    with patch("traffic_detection_kpi.tracking.DeepSort") as MockDS:
        mock_ds = MagicMock()
        mock_ds.update_tracks.return_value = [mock_track]
        MockDS.return_value = mock_ds

        tracker = DeepSortTracker(config)
        result = tracker.track([], MagicMock())

    assert len(result) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tracking.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `tracking.py`**

Create `traffic_detection_kpi/tracking.py`:

```python
from deep_sort_realtime.deepsort_tracker import DeepSort

from traffic_detection_kpi import Detection, TrackedObject
from traffic_detection_kpi.config import TrackerConfig, COCO_CLASS_MAP


class DeepSortTracker:
    def __init__(self, config: TrackerConfig):
        self.tracker = DeepSort(
            max_age=config.max_age,
            n_init=config.n_init,
            nms_max_overlap=0.3,
            max_cosine_distance=config.max_cosine_distance,
            nn_budget=None,
            override_track_class=None,
            embedder=config.embedder,
            half=True,
            bgr=True,
        )

    def track(self, detections: list[Detection], frame) -> list[TrackedObject]:
        # Convert Detection to DeepSort format: ([x, y, w, h], confidence, class_name)
        ds_detections = []
        for det in detections:
            ds_detections.append((list(det.bbox), det.confidence, det.class_name))

        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        result = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Get class info from the track's detection
            det = track.get_det_class()
            class_name = str(det) if det is not None else "unknown"
            class_id = COCO_CLASS_MAP.get(class_name, -1)

            result.append(TrackedObject(
                track_id=track.track_id,
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=class_name,
                center=center,
            ))
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracking.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add traffic_detection_kpi/tracking.py tests/test_tracking.py
git commit -m "feat: DeepSort tracking wrapper with bbox conversion"
```

---

### Task 7: Reporting module

**Files:**
- Create: `traffic_detection_kpi/reporting.py`
- Create: `tests/test_reporting.py`

- [ ] **Step 1: Write failing tests for reporting**

Create `tests/test_reporting.py`:

```python
import json
from pathlib import Path

from traffic_detection_kpi import LaneMetrics, MetricsResult


def _sample_result() -> MetricsResult:
    return MetricsResult(
        video_path="test.mp4",
        total_frames=300,
        duration_seconds=10.0,
        fps=30,
        lanes={
            "Lane 1": LaneMetrics(
                throughput_total=5,
                throughput_rate_avg=0.5,
                vehicle_counts={"car": 3, "truck": 2},
                queue_length_timeseries=[1, 2, 3, 2, 1, 2, 3, 2, 1, 0],
                avg_dwell_time_timeseries=[0.5, 1.0, 1.5, 1.2, 0.8, 0.9, 1.1, 1.3, 0.7, 0.0],
            ),
            "Lane 2": LaneMetrics(
                throughput_total=3,
                throughput_rate_avg=0.3,
                vehicle_counts={"car": 2, "motorcycle": 1},
                queue_length_timeseries=[0, 1, 1, 2, 1, 0, 1, 1, 0, 0],
                avg_dwell_time_timeseries=[0.0, 0.5, 0.8, 1.0, 0.6, 0.0, 0.4, 0.7, 0.0, 0.0],
            ),
        },
    )


def test_generates_json(tmp_path):
    from traffic_detection_kpi.reporting import ReportGenerator
    gen = ReportGenerator(str(tmp_path))
    gen.generate(_sample_result())
    json_path = tmp_path / "metrics.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["fps"] == 30
    assert data["lanes"]["Lane 1"]["throughput_total"] == 5


def test_generates_chart_pngs(tmp_path):
    from traffic_detection_kpi.reporting import ReportGenerator
    gen = ReportGenerator(str(tmp_path))
    gen.generate(_sample_result())
    charts_dir = tmp_path / "charts"
    assert (charts_dir / "throughput_by_lane.png").exists()
    assert (charts_dir / "queue_length_over_time.png").exists()
    assert (charts_dir / "dwell_time_over_time.png").exists()
    assert (charts_dir / "vehicle_class_breakdown.png").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reporting.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `reporting.py`**

Create `traffic_detection_kpi/reporting.py`:

```python
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from traffic_detection_kpi import MetricsResult


class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / "charts"

    def generate(self, result: MetricsResult):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(result)
        self._chart_throughput(result)
        self._chart_queue_length(result)
        self._chart_dwell_time(result)
        self._chart_vehicle_classes(result)

    def _write_json(self, result: MetricsResult):
        data = asdict(result)
        path = self.output_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _chart_throughput(self, result: MetricsResult):
        sns.set_theme()
        names = list(result.lanes.keys())
        totals = [m.throughput_total for m in result.lanes.values()]
        fig, ax = plt.subplots()
        ax.bar(names, totals)
        ax.set_xlabel("Lane")
        ax.set_ylabel("Total vehicles")
        ax.set_title("Throughput by Lane")
        fig.savefig(self.charts_dir / "throughput_by_lane.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _chart_queue_length(self, result: MetricsResult):
        sns.set_theme()
        fig, ax = plt.subplots()
        for name, metrics in result.lanes.items():
            ax.plot(metrics.queue_length_timeseries, label=name)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Queue length")
        ax.set_title("Queue Length Over Time")
        ax.legend()
        fig.savefig(self.charts_dir / "queue_length_over_time.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _chart_dwell_time(self, result: MetricsResult):
        sns.set_theme()
        fig, ax = plt.subplots()
        for name, metrics in result.lanes.items():
            ax.plot(metrics.avg_dwell_time_timeseries, label=name)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Average dwell time (s)")
        ax.set_title("Dwell Time Over Time")
        ax.legend()
        fig.savefig(self.charts_dir / "dwell_time_over_time.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _chart_vehicle_classes(self, result: MetricsResult):
        sns.set_theme()
        # Aggregate across all lanes
        totals: dict[str, int] = {}
        for metrics in result.lanes.values():
            for cls, count in metrics.vehicle_counts.items():
                totals[cls] = totals.get(cls, 0) + count
        if not totals:
            # Create empty chart
            fig, ax = plt.subplots()
            ax.set_title("Vehicle Class Breakdown")
            fig.savefig(self.charts_dir / "vehicle_class_breakdown.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            return
        fig, ax = plt.subplots()
        ax.bar(list(totals.keys()), list(totals.values()))
        ax.set_xlabel("Vehicle class")
        ax.set_ylabel("Count")
        ax.set_title("Vehicle Class Breakdown")
        fig.savefig(self.charts_dir / "vehicle_class_breakdown.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reporting.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add traffic_detection_kpi/reporting.py tests/test_reporting.py
git commit -m "feat: report generation with JSON and matplotlib charts"
```

---

## Chunk 3: Pipeline & CLI

### Task 8: Pipeline module

**Files:**
- Create: `traffic_detection_kpi/pipeline.py`

- [ ] **Step 1: Implement `pipeline.py`**

Create `traffic_detection_kpi/pipeline.py`:

```python
import logging

import cv2

from traffic_detection_kpi.config import Config
from traffic_detection_kpi.detection import YoloDetector
from traffic_detection_kpi.tracking import DeepSortTracker
from traffic_detection_kpi.lanes import LaneZone
from traffic_detection_kpi.metrics import MetricsCollector
from traffic_detection_kpi.reporting import ReportGenerator

logger = logging.getLogger(__name__)


class VideoPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.detector = YoloDetector(
            model_path=config.model.path,
            confidence=config.model.confidence,
            class_filter=config.model.classes,
        )
        self.tracker = DeepSortTracker(config.tracker)
        self.lanes = [LaneZone(l.name, l.polygon) for l in config.lanes]
        self.reporter = ReportGenerator(config.output_dir)
        self._fps: int | None = None

    def run(self):
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.config.video_path}")

        self._fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        metrics = MetricsCollector(
            lane_names=[l.name for l in self.lanes],
            video_fps=self._fps,
            max_age=self.config.tracker.max_age,
        )

        frame_num = 0
        consecutive_failures = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures < 5 and frame_num < total_frames - 1:
                    logger.warning(f"Corrupted frame at {frame_num}, seeking past it")
                    frame_num += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    continue
                break
            consecutive_failures = 0
            frame_num += 1

            detections = self.detector.detect(frame)
            tracked = self.tracker.track(detections, frame)
            assignments = LaneZone.classify(self.lanes, tracked)
            metrics.update(assignments)

            if frame_num % 100 == 0:
                logger.info(f"Processed {frame_num}/{total_frames} frames")

        cap.release()
        logger.info(f"Processing complete: {frame_num} frames")

        result = metrics.finalize(
            video_path=self.config.video_path,
            total_frames=frame_num,
        )
        self.reporter.generate(result)
        logger.info(f"Report saved to {self.config.output_dir}")
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from traffic_detection_kpi.pipeline import VideoPipeline; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add traffic_detection_kpi/pipeline.py
git commit -m "feat: video pipeline orchestrating detection through reporting"
```

---

### Task 9: CLI entry point

**Files:**
- Create: `traffic_detection_kpi/__main__.py`

- [ ] **Step 1: Implement `__main__.py`**

Create `traffic_detection_kpi/__main__.py`:

```python
import argparse
import logging
import sys

from traffic_detection_kpi.config import load_config
from traffic_detection_kpi.pipeline import VideoPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Detection KPI — offline video analytics"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    pipeline = VideoPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

Run: `python -m traffic_detection_kpi --help`
Expected: Shows usage with `--config` and `--verbose` options

- [ ] **Step 3: Commit**

```bash
git add traffic_detection_kpi/__main__.py
git commit -m "feat: CLI entry point with argparse"
```

---

### Task 10: Integration test

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Create `tests/conftest.py` for integration marker**

Create `tests/conftest.py`:

```python
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", default=False, help="Run integration tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="Need --run-integration to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
```

- [ ] **Step 2: Write integration test**

Create `tests/test_pipeline.py`:

```python
"""Integration test — requires a video file and YOLO model to run.

Skip by default; run with: pytest tests/test_pipeline.py -v --run-integration
"""
import json
import pytest
from pathlib import Path


@pytest.mark.integration
def test_full_pipeline(tmp_path):
    """Run the full pipeline on a short video and verify output structure."""
    from traffic_detection_kpi.config import load_config
    from traffic_detection_kpi.pipeline import VideoPipeline

    # Write a test config pointing to the example video
    config_content = f"""\
video_path: "../trafficData/four lanes.mp4"
output_dir: "{tmp_path}"
model:
  path: "yolo11m.pt"
  confidence: 0.2
  classes: [car, motorcycle, bus, truck]
tracker:
  type: deepsort
  max_age: 20
  n_init: 2
  max_cosine_distance: 0.8
  embedder: mobilenet
lanes:
  - name: "Lane 1"
    polygon: [[300, 570], [750, 570], [650, 150], [500, 150]]
  - name: "Lane 2"
    polygon: [[610, 550], [750, 550], [650, 150], [596, 150]]
  - name: "Lane 3"
    polygon: [[770, 550], [900, 550], [770, 200], [670, 200]]
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    config = load_config(str(config_path))
    pipeline = VideoPipeline(config)
    pipeline.run()

    # Verify outputs exist
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "charts" / "throughput_by_lane.png").exists()
    assert (tmp_path / "charts" / "queue_length_over_time.png").exists()
    assert (tmp_path / "charts" / "dwell_time_over_time.png").exists()
    assert (tmp_path / "charts" / "vehicle_class_breakdown.png").exists()

    # Verify JSON structure
    data = json.loads((tmp_path / "metrics.json").read_text())
    assert "lanes" in data
    assert "Lane 1" in data["lanes"]
    assert "throughput_total" in data["lanes"]["Lane 1"]
    assert "queue_length_timeseries" in data["lanes"]["Lane 1"]
```

- [ ] **Step 3: Verify integration test is skipped by default**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 1 test skipped ("Need --run-integration to run")

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_pipeline.py
git commit -m "feat: integration test for full pipeline with --run-integration flag"
```

---

### Task 11: Run all unit tests

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: All unit tests PASS (config: 6, lanes: 5, metrics: 7, detection: 2, tracking: 2, reporting: 2 = 24 tests), integration test skipped

- [ ] **Step 2: Final commit if any fixes needed**

If any tests fail, fix and commit with descriptive message.

---

### Task 12: Clean up old prototype files

- [ ] **Step 1: Remove old prototype scripts**

```bash
git rm bytetrack_simple_counter.py Class_refactor_DeepSort_model_video_processing_non_realtime.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: remove prototype scripts replaced by modular package"
```
