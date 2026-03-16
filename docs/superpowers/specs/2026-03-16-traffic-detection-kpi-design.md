# Traffic Detection KPI — System Design

## Overview

Offline video analytics system that processes recorded traffic video to compute per-lane KPIs: throughput, queue length, dwell time, and vehicle class breakdown. Results are exported as JSON and matplotlib charts.

## Decisions

- **Deployment:** Offline analytics on recorded video only
- **Lane definition:** User provides lane polygons in a YAML config file per video
- **Output formats:** JSON + matplotlib charts (PNGs)
- **Tracker:** DeepSort (accurate re-identification)
- **Detector:** YOLO11m (medium — good accuracy, reasonable speed for offline use; upgrade from nano in prototype)
- **Vehicle classes:** car, motorcycle, bus, truck (no pedestrians — removed intentionally; can be re-enabled via config)
- **Lane count:** Configurable (N lanes)
- **Pub/Sub system:** Removed. The prototype's Publisher/Subscriber was scaffolding for real-time use. Not needed for offline analytics.
- **Annotated video output:** Not included. The system produces metrics and charts only. Annotated video can be added later as an optional output if needed.
- **Frame processing:** Every frame is processed. DeepSort requires continuous frames for reliable re-identification. No frame skipping.

## Configuration

YAML config file per video/camera (e.g. `configs/four_lanes.yaml`):

```yaml
video_path: "../trafficData/four lanes.mp4"
output_dir: "./output"
model:
  path: "yolo11m.pt"
  confidence: 0.2
  classes: [car, motorcycle, bus, truck]  # COCO class names (mapped to IDs internally)
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

`config.py` validates the schema (required fields, polygon format, non-self-intersecting polygons, valid class names) and returns a typed dataclass. Class name strings are mapped to COCO integer IDs internally:

| Name       | COCO ID |
|------------|---------|
| car        | 2       |
| motorcycle | 3       |
| bus        | 5       |
| truck      | 7       |

## Package Structure

```
traffic_detection_kpi/
  __init__.py
  __main__.py            # CLI entry point: parse args, load config, run pipeline
  config.py              # YAML loader, schema validation, Config dataclass
  detection.py           # YoloDetector: wraps YOLO model, returns standardized detections
  tracking.py            # DeepSortTracker: wraps DeepSort, maps detections to tracked objects
  lanes.py               # LaneZone class: polygon definition, point-in-polygon checks
  metrics.py             # MetricsCollector: per-lane queue length, throughput, dwell times
  pipeline.py            # VideoPipeline: reads frames, orchestrates detect->track->classify->metrics
  reporting.py           # ReportGenerator: JSON export + matplotlib charts
configs/
  example.yaml           # Example config file
tests/
  test_config.py
  test_detection.py
  test_tracking.py
  test_lanes.py
  test_metrics.py
  test_pipeline.py
  test_reporting.py
pyproject.toml
```

## Data Flow

```
Video Frame
  -> YoloDetector.detect(frame) -> list[Detection]        (bbox in xywh format)
  -> DeepSortTracker.track(detections, frame) -> list[TrackedObject]  (bbox in ltrb format, center computed)
  -> LaneZone.classify(tracked_objects) -> dict[lane_name -> list[TrackedObject]]
  -> MetricsCollector.update(lane_assignments) -> updates running KPIs
  -> (after all frames) MetricsCollector.finalize() -> MetricsResult
  -> ReportGenerator.generate(metrics_result) -> JSON file + chart PNGs
```

**Bbox format conversion:** `Detection` uses xywh (what YOLO returns). `DeepSortTracker.track()` converts xywh to the format DeepSort expects for `update_tracks()`, then converts DeepSort's output (`to_ltrb()`) to populate `TrackedObject.bbox` in ltrb format. All downstream consumers (LaneZone, MetricsCollector) receive ltrb.

Each module only depends on the one before it in the chain — no circular dependencies.

## Key Classes & Interfaces

### Shared Data Types

```python
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
    center: tuple[int, int]  # midpoint of ltrb bbox, computed by DeepSortTracker

@dataclass
class LaneMetrics:
    throughput_total: int
    throughput_rate_avg: float  # vehicles per second, averaged over entire video
    vehicle_counts: dict[str, int]  # class_name -> count
    queue_length_timeseries: list[int]  # sampled once per second
    avg_dwell_time_timeseries: list[float]  # sampled once per second

@dataclass
class MetricsResult:
    video_path: str
    total_frames: int
    duration_seconds: float
    fps: int
    lanes: dict[str, LaneMetrics]  # lane_name -> LaneMetrics
```

### YoloDetector

- `__init__(model_path, confidence, class_filter)` — loads model, stores allowed COCO class IDs
- `detect(frame) -> list[Detection]` — runs inference, filters by allowed classes

### DeepSortTracker

- `__init__(config)` — initializes DeepSort with tracker params from config
- `track(detections, frame) -> list[TrackedObject]` — converts Detection xywh to DeepSort input format, runs tracking, converts output to TrackedObject with ltrb bbox and center = `((x1+x2)//2, (y1+y2)//2)`

### LaneZone

- `__init__(name, polygon_coords)` — creates a Shapely Polygon
- `contains(point) -> bool`
- Class method `classify(lanes, tracked_objects) -> dict[str, list[TrackedObject]]` — assigns each object to a lane based on its center point. **First-match wins:** lanes are checked in config order; an object is assigned to the first lane whose polygon contains its center. Objects matching no lane are excluded.

### MetricsCollector

- `__init__(lane_names, video_fps)`
- `update(lane_assignments)` — called per frame; updates queue counts, dwell time tracking, throughput counters
- `finalize() -> MetricsResult` — computes final KPIs
- Internally tracks: per-lane current queue length, per-track frame count (for dwell time), per-lane total throughput, and per-second snapshots for time-series charts
- **Track pruning:** tracks not seen for `tracker.max_age` frames are removed from the internal dwell time dictionary to prevent unbounded memory growth

### ReportGenerator

- `__init__(output_dir)`
- `generate(metrics_result: MetricsResult)` — writes `metrics.json` + chart PNGs (throughput bar chart, queue length over time, dwell time over time, vehicle class breakdown)

## Pipeline Orchestration

```python
class VideoPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.detector = YoloDetector(config.model)
        self.tracker = DeepSortTracker(config.tracker)
        self.lanes = [LaneZone(l.name, l.polygon) for l in config.lanes]
        self.metrics = MetricsCollector(
            lane_names=[l.name for l in config.lanes],
            video_fps=self._get_fps(config.video_path)
        )
        self.reporter = ReportGenerator(config.output_dir)

    def _get_fps(self, video_path: str) -> int:
        """Read FPS from video file metadata via cv2.CAP_PROP_FPS."""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps

    def run(self):
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.config.video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            detections = self.detector.detect(frame)
            tracked = self.tracker.track(detections, frame)
            assignments = LaneZone.classify(self.lanes, tracked)
            self.metrics.update(assignments)
            if frame_num % 100 == 0:
                logger.info(f"Processed {frame_num}/{total_frames} frames")
        cap.release()
        result = self.metrics.finalize()
        self.reporter.generate(result)
```

## CLI

```bash
python -m traffic_detection_kpi --config configs/four_lanes.yaml
```

`__main__.py` parses args, loads the config, creates `VideoPipeline`, and calls `run()`. Uses Python's `logging` module (INFO level by default, DEBUG with `--verbose`).

## Error Handling

- **Video file not found / unreadable:** `VideoPipeline.run()` raises `RuntimeError` with descriptive message
- **Invalid config:** `config.py` raises `ValueError` with details on which field failed validation (missing fields, invalid polygon, unknown class name)
- **Model file not found:** `YoloDetector.__init__()` raises `FileNotFoundError`
- **Corrupted frames:** `cap.read()` returning `(False, None)` mid-video is logged as a warning; processing continues with next frame rather than stopping

## Metrics Computation

### Throughput

A vehicle counts toward a lane's throughput once it has been present in that lane for at least `video_fps` consecutive frames (1 second). This filters out momentary misdetections. Throughput rate = total count / elapsed seconds.

### Queue Length

Number of tracked objects currently inside a lane polygon on a given frame. Sampled once per second for the time-series output.

### Dwell Time

For each tracked object in a lane, dwell time = number of frames present / video_fps (in seconds). This measures how long a vehicle is visible within a lane zone — not whether it is stationary. Per-lane average dwell time is computed each frame and sampled once per second for time-series.

### Vehicle Class Breakdown

Per-lane count of each vehicle type (car, motorcycle, bus, truck), only counting vehicles that pass the 1-second threshold.

## Output

### Directory Structure

```
output/
  metrics.json
  charts/
    throughput_by_lane.png
    queue_length_over_time.png
    dwell_time_over_time.png
    vehicle_class_breakdown.png
```

### metrics.json Structure

```json
{
  "video_path": "...",
  "total_frames": 2040,
  "duration_seconds": 68.0,
  "fps": 30,
  "lanes": {
    "Lane 1": {
      "throughput_total": 42,
      "throughput_rate_avg": 0.62,
      "vehicle_counts": {"car": 35, "motorcycle": 3, "bus": 2, "truck": 2},
      "queue_length_timeseries": [2, 3, 1],
      "avg_dwell_time_timeseries": [1.2, 2.1, 0.8]
    }
  }
}
```

## Testing Strategy

- **Unit tests** for each module using pytest:
  - `test_config.py` — valid/invalid YAML parsing, class name to COCO ID mapping
  - `test_detection.py` — mock YOLO model, verify Detection output format and class filtering
  - `test_tracking.py` — mock DeepSort, verify xywh-to-ltrb conversion and center computation
  - `test_lanes.py` — point-in-polygon with known coordinates, first-match-wins ordering, objects outside all lanes
  - `test_metrics.py` — synthetic frame-by-frame updates, verify throughput threshold, dwell time accumulation, per-second sampling, track pruning
  - `test_reporting.py` — verify JSON structure, chart file creation
- **Integration test** (`test_pipeline.py`): run the full pipeline on a short test video clip (5-10 seconds) with known lane polygons and verify output metrics are within expected ranges

## Dependencies

Specified in `pyproject.toml`:

- ultralytics (YOLO)
- deep-sort-realtime (DeepSort tracker)
- opencv-python (video I/O, frame processing)
- shapely (point-in-polygon for lane classification)
- matplotlib, seaborn (chart generation)
- pyyaml (config loading)
- pytest (dev dependency, testing)
