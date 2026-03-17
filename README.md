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

### Options

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML config file (required) |
| `--youtube URL` | YouTube live stream URL |
| `--rtsp URL` | RTSP or RTMP stream URL |
| `--verbose` | Enable debug logging |

For live streams, press **Ctrl+C** to stop processing. The pipeline will finish the current frame and generate the report on all data collected so far.
