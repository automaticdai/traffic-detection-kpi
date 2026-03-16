# Traffic Detection KPI — System Design

## Overview

Offline video analytics system that processes recorded traffic video to compute per-lane KPIs: throughput, queue length, wait time, and vehicle class breakdown. Results are exported as JSON and matplotlib charts.

## Decisions

- **Deployment:** Offline analytics on recorded video only
- **Lane definition:** User provides lane polygons in a YAML config file per video
- **Output formats:** JSON + matplotlib charts (PNGs)
- **Tracker:** DeepSort (accurate re-identification)
- **Detector:** YOLO11m (medium — good accuracy, reasonable speed for offline use)
- **Vehicle classes:** car, motorcycle, bus, truck (no pedestrians)
- **Lane count:** Configurable (N lanes)

## Configuration

YAML config file per video/camera (e.g. `configs/four_lanes.yaml`):

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

A config loader validates the schema (required fields, polygon format, valid class names) and returns a typed dataclass.

## Package Structure

```
traffic_detection_kpi/
  __init__.py
  __main__.py            # CLI entry point: parse args, load config, run pipeline
  config.py              # YAML loader, schema validation, Config dataclass
  detection.py           # YoloDetector: wraps YOLO model, returns standardized detections
  tracking.py            # DeepSortTracker: wraps DeepSort, maps detections to tracked objects
  lanes.py               # LaneZone class: polygon definition, point-in-polygon checks
  metrics.py             # MetricsCollector: per-lane queue length, throughput, wait times
  pipeline.py            # VideoPipeline: reads frames, orchestrates detect->track->classify->metrics
  reporting.py           # ReportGenerator: JSON export + matplotlib charts
configs/
  example.yaml           # Example config file
```

## Data Flow

```
Video Frame
  -> YoloDetector.detect(frame) -> list[Detection(bbox, class_id, confidence)]
  -> DeepSortTracker.track(detections, frame) -> list[TrackedObject(track_id, bbox, class_id)]
  -> LaneZone.classify(tracked_objects) -> dict[lane_name -> list[TrackedObject]]
  -> MetricsCollector.update(lane_assignments) -> updates running KPIs
  -> (after all frames) MetricsCollector.finalize() -> MetricsResult
  -> ReportGenerator.generate(metrics_result) -> JSON file + chart PNGs
```

Each module only depends on the one before it in the chain — no circular dependencies.

## Key Classes & Interfaces

### Shared Data Types

```python
@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x1, y1, w, h
    class_id: int
    class_name: str
    confidence: float

@dataclass
class TrackedObject:
    track_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (ltrb)
    class_id: int
    class_name: str
    center: tuple[int, int]
```

### YoloDetector

- `__init__(model_path, confidence, class_filter)` — loads model, stores allowed classes
- `detect(frame) -> list[Detection]`

### DeepSortTracker

- `__init__(config)` — initializes DeepSort with tracker params from config
- `track(detections, frame) -> list[TrackedObject]`

### LaneZone

- `__init__(name, polygon_coords)` — creates a Shapely Polygon
- `contains(point) -> bool`
- Class method `classify(lanes, tracked_objects) -> dict[str, list[TrackedObject]]` — assigns each object to a lane (or none)

### MetricsCollector

- `__init__(lane_names, video_fps)`
- `update(lane_assignments)` — called per frame; updates queue counts, wait time tracking, throughput counters
- `finalize() -> MetricsResult` — computes final KPIs
- Internally tracks: per-lane current queue length, per-track frame count (for wait time), per-lane total throughput, and per-second snapshots for time-series charts

### MetricsResult

- Per-lane totals: throughput, vehicle counts by class
- Time-series: queue length over time, average wait time over time (sampled every second)

### ReportGenerator

- `__init__(output_dir)`
- `generate(metrics_result)` — writes `metrics.json` + chart PNGs

## Pipeline Orchestration

```python
class VideoPipeline:
    def __init__(self, config: Config):
        self.detector = YoloDetector(config.model)
        self.tracker = DeepSortTracker(config.tracker)
        self.lanes = [LaneZone(l.name, l.polygon) for l in config.lanes]
        self.metrics = MetricsCollector(
            lane_names=[l.name for l in config.lanes],
            video_fps=self._get_fps(config.video_path)
        )
        self.reporter = ReportGenerator(config.output_dir)

    def run(self):
        cap = cv2.VideoCapture(config.video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = self.detector.detect(frame)
            tracked = self.tracker.track(detections, frame)
            assignments = LaneZone.classify(self.lanes, tracked)
            self.metrics.update(assignments)
        cap.release()
        result = self.metrics.finalize()
        self.reporter.generate(result)
```

## CLI

```bash
python -m traffic_detection_kpi --config configs/four_lanes.yaml
```

`__main__.py` parses args, loads the config, creates `VideoPipeline`, and calls `run()`. Progress is printed to stdout (frame count, percentage complete).

## Metrics Computation

### Throughput

A vehicle counts toward a lane's throughput once it has been present in that lane for at least `video_fps` consecutive frames (1 second). This filters out momentary misdetections. Throughput rate = total count / elapsed seconds.

### Queue Length

Number of tracked objects currently inside a lane polygon on a given frame. Sampled once per second for the time-series output.

### Wait Time

For each tracked object in a lane, wait time = number of frames present / video_fps (in seconds). Per-lane average wait time is computed each frame and sampled once per second for time-series.

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
    wait_time_over_time.png
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
      "throughput_per_second": 0.62,
      "vehicle_counts": {"car": 35, "motorcycle": 3, "bus": 2, "truck": 2},
      "queue_length_timeseries": [2, 3, 1],
      "avg_wait_time_timeseries": [1.2, 2.1, 0.8]
    }
  }
}
```

## Dependencies

- ultralytics (YOLO)
- deep-sort-realtime (DeepSort tracker)
- opencv-python (video I/O, frame processing)
- shapely (point-in-polygon for lane classification)
- matplotlib, seaborn (chart generation)
- pyyaml (config loading)
