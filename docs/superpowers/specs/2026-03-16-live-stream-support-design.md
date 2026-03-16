# Live Stream Support Design

**Date:** 2026-03-16
**Status:** Approved
**Scope:** Add YouTube and RTSP/RTMP live stream support to the traffic detection pipeline

## Problem

The pipeline currently only processes local video files via `cv2.VideoCapture(path)`. Traffic cameras frequently broadcast as live YouTube streams or RTSP/RTMP feeds. Users need to analyze these sources in real time without downloading them first.

## Requirements

- Process live YouTube streams by URL
- Process RTSP/RTMP streams by URL
- Graceful shutdown on Ctrl+C with report generation on collected data
- Preserve existing file-based workflow unchanged
- Handle transient network failures with retry logic

## Non-Requirements

- WebRTC support (deferred to a future iteration)
- Duration-based windowing (run-until-interrupted only)
- Annotated/overlaid video output

## Design

### VideoSource Protocol

A new module `traffic_detection_kpi/source.py` introduces a `VideoSource` protocol that all video inputs implement:

```python
class VideoSource(Protocol):
    def read(self) -> tuple[bool, numpy.ndarray | None]:
        """Read the next frame. Returns (success, frame)."""
        ...

    @property
    def fps(self) -> int:
        """Frames per second of the source."""
        ...

    @property
    def is_live(self) -> bool:
        """True for live streams, False for recorded files."""
        ...

    def release(self) -> None:
        """Release underlying resources."""
        ...
```

### Source Implementations

**FileSource**
- Wraps `cv2.VideoCapture` on a local file path
- `is_live = False`
- `fps` read from `CAP_PROP_FPS`
- Preserves current behavior exactly

**YouTubeSource**
- Uses yt-dlp Python API to resolve a YouTube URL to a direct stream URL
- Selects the best available video stream (preferring 720p or lower to balance quality and bandwidth)
- Passes the resolved URL to `cv2.VideoCapture`
- `is_live = True`
- `fps` from yt-dlp metadata, fallback to OpenCV's reported FPS, fallback to 30
- Validates URL and stream availability at construction time; raises clear error on failure

**RtspSource**
- Passes RTSP or RTMP URL directly to `cv2.VideoCapture`
- `is_live = True`
- `fps` from OpenCV's `CAP_PROP_FPS`, fallback to 30
- Validates stream is reachable at construction time via `cap.isOpened()`

### Reconnection Logic

Live sources (`is_live = True`) implement retry logic on transient read failures:
- On `read()` returning `(False, None)`, retry up to 5 times with 1-second backoff
- After 5 consecutive failures, `read()` returns `(False, None)` permanently, signaling the pipeline to shut down and finalize
- File sources do not retry (current behavior preserved)

### Pipeline Changes

**`pipeline.py`:**
- Constructor accepts a `VideoSource` instead of building `cv2.VideoCapture` internally
- For live sources (`is_live = True`):
  - Total frame count is unavailable; progress logging uses elapsed time instead
  - Registers a `SIGINT` signal handler that sets an internal `_shutdown` flag
  - Each iteration of the frame loop checks `_shutdown`; when set, the loop exits cleanly
  - After loop exit, proceeds to `finalize()` and reporting as normal
- For file sources: behavior is unchanged from current implementation

### Configuration Changes

**`config.py`:**
- `video_path` becomes optional (can be `None`)
- Validation ensures at least one source is provided (either `video_path` in config or a CLI flag)

**`__main__.py`:**
- New CLI flags:
  - `--youtube <url>` — YouTube stream URL
  - `--rtsp <url>` — RTSP or RTMP stream URL
- Source resolution priority: `--youtube` or `--rtsp` flag overrides config's `video_path`
- Validation: at most one source flag can be provided
- Factory logic constructs the appropriate `VideoSource` based on resolved input

### New Dependency

- `yt-dlp` — added to `pyproject.toml` dependencies

No new dependencies needed for RTSP/RTMP (OpenCV handles natively).

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Invalid YouTube URL | yt-dlp raises error at construction; clear message to user |
| Private/geo-blocked video | yt-dlp raises error at construction; clear message to user |
| Unreachable RTSP stream | `cap.isOpened()` returns False; clear error at construction |
| Transient network drop | Retry up to 5 times with 1s backoff, then finalize |
| Ctrl+C during processing | Set shutdown flag, finish current frame, finalize and report |

### Testing

**New unit tests in `tests/test_source.py`:**
- `FileSource`: mock `cv2.VideoCapture`, verify `read()`, `fps`, `is_live=False`, `release()`
- `YouTubeSource`: mock yt-dlp and `cv2.VideoCapture`, verify URL resolution, `is_live=True`, error cases (invalid URL, private video)
- `RtspSource`: mock `cv2.VideoCapture`, verify `is_live=True`, error on unreachable stream
- Retry logic: mock transient failures, verify retry count and backoff
- Graceful shutdown: verify `_shutdown` flag stops frame loop

**Existing tests:** No changes required. `FileSource` wraps the same OpenCV behavior the pipeline currently uses.

### Module Dependency Graph

```
__main__.py
├── config.py          (load YAML, validate)
├── source.py          (construct VideoSource from CLI args + config)
│   ├── FileSource     (cv2)
│   ├── YouTubeSource  (yt-dlp + cv2)
│   └── RtspSource     (cv2)
└── pipeline.py        (accepts VideoSource)
    ├── detection.py
    ├── tracking.py
    ├── lanes.py
    ├── metrics.py
    └── reporting.py
```

### Files Changed

| File | Change |
|------|--------|
| `traffic_detection_kpi/source.py` | **New** — VideoSource protocol + FileSource, YouTubeSource, RtspSource |
| `traffic_detection_kpi/pipeline.py` | Accept VideoSource, graceful shutdown for live sources |
| `traffic_detection_kpi/config.py` | Make `video_path` optional |
| `traffic_detection_kpi/__main__.py` | Add `--youtube` and `--rtsp` CLI flags, source factory |
| `pyproject.toml` | Add `yt-dlp` dependency |
| `tests/test_source.py` | **New** — unit tests for all sources |
| `tests/test_pipeline.py` | Update to use VideoSource (if needed) |
