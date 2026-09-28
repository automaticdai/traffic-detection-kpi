# Traffic Detection KPI

Detect and measure key traffic metrics from video feeds using computer vision.

## Supported Inputs

- Live video stream from YouTube
- Live video stream via RTSP / RTMP
- Recorded video files

## Features

- **Lane regions** — lanes are polygons defined in the config, drawn and adjusted with the interactive lane editor
- **Vehicle detection** — YOLO detects vehicles in each frame; DeepSORT tracks them across frames
- **Per-lane metrics** — throughput, queue length, and vehicle class breakdown
- **Per-vehicle metrics** — dwell time within each lane
- **Output** — displays results on screen and saves them to a JSON file with charts

## Metrics

Written to `output/metrics.json`, with charts in `output/charts/`.

| Metric | Definition |
|--------|------------|
| `throughput_total` | Completed passages through the lane. A tracked vehicle is counted when it *leaves* the lane — moving to another lane, disappearing, or the run ending — provided it stayed at least 0.2 s. Counted at most once per lane, so bounding-box jitter across a boundary cannot inflate it. A vehicle that changes lanes counts once in each lane it traverses. |
| `throughput_rate_avg` | `throughput_total / duration_seconds`. For live streams `duration_seconds` is wall-clock elapsed, not frame count ÷ fps, since a stream that outpaces inference drops frames. |
| `queue_length_timeseries` | Vehicles present in the lane, sampled once per second. |
| `avg_dwell_time_timeseries` | Mean time vehicles have spent in *their current lane*, sampled once per second. Resets when a vehicle changes lane. |
| `vehicle_counts` | Per-class breakdown of the vehicles counted in `throughput_total`. |

## Usage

### Recorded video file

```bash
traffic-kpi --config config.yaml
```

Where `config.yaml` contains `video_path: "path/to/video.mp4"`.

### YouTube live stream

```bash
traffic-kpi --config config.yaml --youtube "https://www.youtube.com/watch?v=STREAM_ID"
```

### RTSP / RTMP stream

```bash
traffic-kpi --config config.yaml --rtsp "rtsp://camera.example.com/stream"
```

RTMP streams also use the `--rtsp` flag:

```bash
traffic-kpi --config config.yaml --rtsp "rtmp://camera.example.com/live/stream"
```

### Live GUI overlay

Add `--show` to any command to see detections, lane regions, and live metrics:

```bash
traffic-kpi --config config.yaml --show
traffic-kpi --config config.yaml --youtube "https://www.youtube.com/watch?v=STREAM_ID" --show
```

Press **q** to quit the overlay window, or **Ctrl+C** to stop the pipeline.

### Options

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML config file (required) |
| `--youtube URL` | YouTube live stream URL |
| `--rtsp URL` | RTSP or RTMP stream URL |
| `--show` | Show live GUI overlay with detections and metrics |
| `--verbose` | Enable debug logging |

For live streams, press **Ctrl+C** to stop processing. The pipeline will finish the current frame and generate the report on all data collected so far.

## Lane Editor

Interactive tool for drawing and adjusting lane polygons on a video frame. Changes are saved back to the config file.

### Launch the editor

```bash
# From a local video file
traffic-lane-editor --config config.yaml --video path/to/video.mp4

# From a YouTube stream (grabs first frame)
traffic-lane-editor --config config.yaml --youtube "https://www.youtube.com/watch?v=STREAM_ID"

# From an RTSP/RTMP stream (grabs first frame)
traffic-lane-editor --config config.yaml --rtsp "rtsp://camera.example.com/stream"
```

### Controls

| Action | Input |
|--------|-------|
| Move a vertex | Left click + drag |
| Insert a vertex on an edge | Left click near an edge |
| Select a lane | Left click inside a polygon |
| Delete selected lane | `d` |
| Delete selected vertex | Select vertex, then `x` (min 3 kept) |
| Draw a new lane | `n`, then click to place points, `Enter` to finish |
| Cancel / deselect | `Esc` |
| Quit | `q` (prompts to save if changes were made) |
