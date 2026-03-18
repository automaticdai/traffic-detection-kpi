# Traffic Detection KPI

Detect and measure key traffic metrics from video feeds using computer vision.

## Supported Inputs

- Live video stream from YouTube
- Live video stream via RTSP / RTMP
- Live stream from WebRTC
- Recorded video files

## Features

- **Lane detection** — identifies traffic lanes using a bird's-eye-view (BEV) map
- **Vehicle detection** — detects cars in each frame
- **Per-lane metrics** — measures traffic throughput and vehicle count
- **Per-vehicle metrics** — tracks waiting time for each car
- **Output** — displays results on screen and saves them to a JSON file

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
