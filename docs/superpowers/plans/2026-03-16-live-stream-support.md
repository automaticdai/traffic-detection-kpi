# Live Stream Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add YouTube and RTSP/RTMP live stream support with graceful Ctrl+C shutdown.

**Architecture:** Introduce a `VideoSource` protocol in a new `source.py` module with `FileSource`, `YouTubeSource`, and `RtspSource` implementations. Refactor `VideoPipeline` to consume `VideoSource` instead of raw `cv2.VideoCapture`. Add `--youtube` and `--rtsp` CLI flags.

**Tech Stack:** yt-dlp (YouTube URL resolution), OpenCV (frame reading), Python signal handling (graceful shutdown)

**Spec:** `docs/superpowers/specs/2026-03-16-live-stream-support-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `traffic_detection_kpi/source.py` | Create | `VideoSource` protocol + `FileSource`, `YouTubeSource`, `RtspSource` implementations |
| `tests/test_source.py` | Create | Unit tests for all three sources, retry logic, error cases |
| `traffic_detection_kpi/pipeline.py` | Modify | Accept `VideoSource`, SIGINT graceful shutdown for live sources, remove seek-retry |
| `tests/test_pipeline.py` | Modify | Update integration test to use `FileSource` |
| `traffic_detection_kpi/config.py` | Modify | Make `video_path` optional |
| `tests/test_config.py` | Modify | Add test for config without `video_path` |
| `traffic_detection_kpi/__main__.py` | Modify | Add `--youtube`/`--rtsp` flags, source factory, cross-validation |
| `pyproject.toml` | Modify | Add `yt-dlp` dependency |

---

## Chunk 1: VideoSource Protocol and FileSource

### Task 1: FileSource — tests

**Files:**
- Create: `tests/test_source.py`

- [ ] **Step 1: Write FileSource tests**

```python
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestFileSource:
    def test_read_returns_frame(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = FileSource("video.mp4")
            ret, result = source.read()

        assert ret is True
        assert result is not None
        assert result.shape == (480, 640, 3)

    def test_fps_from_capture(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = FileSource("video.mp4")

        assert source.fps == 25

    def test_is_live_false(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = FileSource("video.mp4")

        assert source.is_live is False

    def test_release_delegates_to_capture(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = FileSource("video.mp4")
            source.release()

        mock_cap.release.assert_called_once()

    def test_raises_on_unopenable_file(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(RuntimeError, match="Cannot open video"):
                FileSource("nonexistent.mp4")

    def test_fps_rounds_not_truncates(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 29.97

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = FileSource("video.mp4")

        assert source.fps == 30  # round(29.97), not int(29.97)=29

    def test_read_eof(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (False, None)

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = FileSource("video.mp4")
            ret, frame = source.read()

        assert ret is False
        assert frame is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'traffic_detection_kpi.source'`

### Task 2: VideoSource protocol and FileSource — implementation

**Files:**
- Create: `traffic_detection_kpi/source.py`

- [ ] **Step 3: Implement VideoSource protocol and FileSource**

```python
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class VideoSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame."""
        ...

    @property
    def fps(self) -> int:
        """Frames per second (rounded to nearest integer)."""
        ...

    @property
    def is_live(self) -> bool:
        """True for live streams, False for recorded files."""
        ...

    @property
    def url(self) -> str:
        """The source path or URL (for provenance in reports)."""
        ...

    def release(self) -> None:
        """Release underlying resources."""
        ...


class FileSource:
    def __init__(self, path: str) -> None:
        self._url = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self._fps = round(self._cap.get(cv2.CAP_PROP_FPS)) or 30

    def read(self) -> tuple[bool, np.ndarray | None]:
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def is_live(self) -> bool:
        return False

    @property
    def url(self) -> str:
        return self._url

    def release(self) -> None:
        self._cap.release()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_source.py::TestFileSource -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add traffic_detection_kpi/source.py tests/test_source.py
git commit -m "feat: add VideoSource protocol and FileSource implementation"
```

---

## Chunk 2: YouTubeSource and RtspSource

### Task 3: YouTubeSource — tests

**Files:**
- Modify: `tests/test_source.py`

- [ ] **Step 6: Write YouTubeSource tests**

Append to `tests/test_source.py`:

```python
class TestYouTubeSource:
    def test_resolves_url_and_reads_frame(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved-stream.example.com/video",
            "fps": 30,
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = YouTubeSource("https://www.youtube.com/watch?v=test")
            ret, result = source.read()

        assert ret is True
        assert result is not None
        assert source.is_live is True

    def test_fps_from_ytdlp_metadata_rounded(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0.0

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved.example.com/video",
            "fps": 29.97,
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = YouTubeSource("https://www.youtube.com/watch?v=test")

        assert source.fps == 30

    def test_fps_fallback_to_opencv(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved.example.com/video",
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = YouTubeSource("https://www.youtube.com/watch?v=test")

        assert source.fps == 25

    def test_fps_fallback_to_default_30(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0.0

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved.example.com/video",
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = YouTubeSource("https://www.youtube.com/watch?v=test")

        assert source.fps == 30

    def test_raises_on_invalid_url(self):
        from traffic_detection_kpi.source import YouTubeSource
        import yt_dlp

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("not found")

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl):
            with pytest.raises(RuntimeError, match="Failed to resolve YouTube URL"):
                YouTubeSource("https://www.youtube.com/watch?v=invalid")

    def test_raises_on_unopenable_stream(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved.example.com/video",
            "fps": 30,
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(RuntimeError, match="Cannot open resolved YouTube stream"):
                YouTubeSource("https://www.youtube.com/watch?v=test")

    def test_retry_on_transient_failure(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (False, None),
            (False, None),
            (True, frame),
        ]

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved.example.com/video",
            "fps": 30,
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap), \
             patch("traffic_detection_kpi.source.time.sleep"):
            source = YouTubeSource("https://www.youtube.com/watch?v=test")
            ret, result = source.read()

        assert ret is True
        assert result is not None

    def test_retry_exhausted_returns_false(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (False, None)

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved.example.com/video",
            "fps": 30,
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap), \
             patch("traffic_detection_kpi.source.time.sleep"):
            source = YouTubeSource("https://www.youtube.com/watch?v=test")
            ret, frame = source.read()

        assert ret is False
        assert frame is None
        assert mock_cap.read.call_count == 6  # 1 initial + 5 retries
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `pytest tests/test_source.py::TestYouTubeSource -v`
Expected: FAIL — `ImportError: cannot import name 'YouTubeSource'`

### Task 4: YouTubeSource — implementation

**Files:**
- Modify: `traffic_detection_kpi/source.py`

- [ ] **Step 8: Implement YouTubeSource**

Add to `source.py` after `FileSource`, adding `import time` and `import yt_dlp` at the top:

```python
import time

import yt_dlp


class YouTubeSource:
    _MAX_RETRIES = 5
    _RETRY_DELAY = 1.0

    def __init__(self, url: str) -> None:
        self._url = url
        stream_url, meta_fps = self._resolve(url)
        self._cap = cv2.VideoCapture(stream_url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open resolved YouTube stream: {url}")
        self._fps = self._determine_fps(meta_fps)

    @staticmethod
    def _resolve(url: str) -> tuple[str, float | None]:
        opts = {
            "format": "best[height<=720][ext=mp4]/best[height<=720]/best",
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            raise RuntimeError(f"Failed to resolve YouTube URL: {e}") from e
        return info["url"], info.get("fps")

    def _determine_fps(self, meta_fps: float | None) -> int:
        if meta_fps:
            return round(meta_fps)
        cv_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if cv_fps > 0:
            return round(cv_fps)
        return 30

    def read(self) -> tuple[bool, np.ndarray | None]:
        ret, frame = self._cap.read()
        if ret:
            return True, frame
        for attempt in range(self._MAX_RETRIES):
            logger.warning("YouTube read failed, retry %d/%d", attempt + 1, self._MAX_RETRIES)
            time.sleep(self._RETRY_DELAY)
            ret, frame = self._cap.read()
            if ret:
                return True, frame
        return False, None

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def is_live(self) -> bool:
        return True

    @property
    def url(self) -> str:
        return self._url

    def release(self) -> None:
        self._cap.release()
```

- [ ] **Step 9: Run YouTubeSource tests**

Run: `pytest tests/test_source.py::TestYouTubeSource -v`
Expected: All 8 tests PASS

- [ ] **Step 10: Commit**

```bash
git add traffic_detection_kpi/source.py tests/test_source.py
git commit -m "feat: add YouTubeSource with yt-dlp resolution and retry logic"
```

### Task 5: RtspSource — tests

**Files:**
- Modify: `tests/test_source.py`

- [ ] **Step 11: Write RtspSource tests**

Append to `tests/test_source.py`:

```python
class TestRtspSource:
    def test_opens_rtsp_url(self):
        from traffic_detection_kpi.source import RtspSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = RtspSource("rtsp://camera.example.com/stream")
            ret, result = source.read()

        assert ret is True
        assert result is not None
        assert source.is_live is True
        assert source.fps == 25

    def test_fps_fallback_to_30(self):
        from traffic_detection_kpi.source import RtspSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0.0

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = RtspSource("rtsp://camera.example.com/stream")

        assert source.fps == 30

    def test_raises_on_unreachable(self):
        from traffic_detection_kpi.source import RtspSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(RuntimeError, match="Cannot open stream"):
                RtspSource("rtsp://bad.example.com/stream")

    def test_rtmp_url(self):
        from traffic_detection_kpi.source import RtspSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = RtspSource("rtmp://camera.example.com/live/stream")

        assert source.is_live is True

    def test_retry_on_transient_failure(self):
        from traffic_detection_kpi.source import RtspSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(False, None), (True, frame)]

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap), \
             patch("traffic_detection_kpi.source.time.sleep"):
            source = RtspSource("rtsp://camera.example.com/stream")
            ret, result = source.read()

        assert ret is True

    def test_retry_exhausted(self):
        from traffic_detection_kpi.source import RtspSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (False, None)

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap), \
             patch("traffic_detection_kpi.source.time.sleep"):
            source = RtspSource("rtsp://camera.example.com/stream")
            ret, frame = source.read()

        assert ret is False
        assert mock_cap.read.call_count == 6  # 1 initial + 5 retries
```

- [ ] **Step 12: Run tests to verify they fail**

Run: `pytest tests/test_source.py::TestRtspSource -v`
Expected: FAIL — `ImportError: cannot import name 'RtspSource'`

### Task 6: RtspSource — implementation

**Files:**
- Modify: `traffic_detection_kpi/source.py`

- [ ] **Step 13: Implement RtspSource**

Add to `source.py` after `YouTubeSource`:

```python
class RtspSource:
    _MAX_RETRIES = 5
    _RETRY_DELAY = 1.0

    def __init__(self, url: str) -> None:
        self._url = url
        self._cap = cv2.VideoCapture(url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {url}")
        cv_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = round(cv_fps) if cv_fps > 0 else 30

    def read(self) -> tuple[bool, np.ndarray | None]:
        ret, frame = self._cap.read()
        if ret:
            return True, frame
        for attempt in range(self._MAX_RETRIES):
            logger.warning("Stream read failed, retry %d/%d", attempt + 1, self._MAX_RETRIES)
            time.sleep(self._RETRY_DELAY)
            ret, frame = self._cap.read()
            if ret:
                return True, frame
        return False, None

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def is_live(self) -> bool:
        return True

    @property
    def url(self) -> str:
        return self._url

    def release(self) -> None:
        self._cap.release()
```

- [ ] **Step 14: Run RtspSource tests**

Run: `pytest tests/test_source.py::TestRtspSource -v`
Expected: All 6 tests PASS

- [ ] **Step 15: Run all source tests**

Run: `pytest tests/test_source.py -v`
Expected: All 21 tests PASS

- [ ] **Step 16: Commit**

```bash
git add traffic_detection_kpi/source.py tests/test_source.py
git commit -m "feat: add RtspSource with retry logic"
```

---

## Chunk 3: Pipeline Refactor

### Task 7: Pipeline accepts VideoSource — tests

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] **Step 17: Add unit test for pipeline with mock VideoSource**

Add a new test before the existing integration test in `tests/test_pipeline.py`:

```python
from unittest.mock import MagicMock, patch
import numpy as np


def test_pipeline_uses_video_source(tmp_path):
    """Pipeline reads from VideoSource instead of cv2.VideoCapture."""
    from traffic_detection_kpi.config import load_config
    from traffic_detection_kpi.pipeline import VideoPipeline

    config_content = f"""\
video_path: "dummy.mp4"
output_dir: "{tmp_path}"
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
    config = load_config(str(config_path))

    mock_source = MagicMock()
    mock_source.fps = 30
    mock_source.is_live = False
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_source.read.side_effect = [(True, frame), (True, frame), (False, None)]

    mock_detector = MagicMock()
    mock_detector.detect.return_value = []

    mock_tracker = MagicMock()
    mock_tracker.track.return_value = []

    pipeline = VideoPipeline(config, source=mock_source)
    pipeline.detector = mock_detector
    pipeline.tracker = mock_tracker
    pipeline.run()

    assert mock_source.read.call_count == 3
    mock_source.release.assert_called_once()
```

- [ ] **Step 18: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_pipeline_uses_video_source -v`
Expected: FAIL — `TypeError: VideoPipeline.__init__() got an unexpected keyword argument 'source'`

### Task 8: Refactor pipeline to accept VideoSource

**Files:**
- Modify: `traffic_detection_kpi/pipeline.py`

- [ ] **Step 19: Refactor pipeline.py**

Replace the full content of `pipeline.py`:

```python
import logging
import signal
import time

import cv2

from traffic_detection_kpi.config import Config
from traffic_detection_kpi.detection import YoloDetector
from traffic_detection_kpi.source import FileSource, VideoSource
from traffic_detection_kpi.tracking import DeepSortTracker
from traffic_detection_kpi.lanes import LaneZone
from traffic_detection_kpi.metrics import MetricsCollector
from traffic_detection_kpi.reporting import ReportGenerator

logger = logging.getLogger(__name__)


class VideoPipeline:
    def __init__(self, config: Config, source: VideoSource | None = None):
        self.config = config
        self.source = source
        self.detector = YoloDetector(
            model_path=config.model.path,
            confidence=config.model.confidence,
            class_filter=config.model.classes,
        )
        self.tracker = DeepSortTracker(config.tracker)
        self.lanes = [LaneZone(l.name, l.polygon) for l in config.lanes]
        self.reporter = ReportGenerator(config.output_dir)

    def run(self):
        source = self.source or FileSource(self.config.video_path)
        shutdown = False

        if source.is_live:
            prev_handler = signal.getsignal(signal.SIGINT)

            def _handle_sigint(signum, frame):
                nonlocal shutdown
                logger.info("Shutdown requested, finishing current frame...")
                shutdown = True

            signal.signal(signal.SIGINT, _handle_sigint)

        try:
            self._run_loop(source, shutdown_check=lambda: shutdown)
        finally:
            if source.is_live:
                signal.signal(signal.SIGINT, prev_handler)
            source.release()

    def _run_loop(self, source: VideoSource, shutdown_check):
        metrics = MetricsCollector(
            lane_names=[l.name for l in self.lanes],
            video_fps=source.fps,
            max_age=self.config.tracker.max_age,
        )

        video_path = source.url
        frame_num = 0
        start_time = time.monotonic()

        while not shutdown_check():
            ret, frame = source.read()
            if not ret:
                break
            frame_num += 1

            detections = self.detector.detect(frame)
            tracked = self.tracker.track(detections, frame)
            assignments = LaneZone.classify(self.lanes, tracked)
            metrics.update(assignments)

            if frame_num % 100 == 0:
                if source.is_live:
                    elapsed = time.monotonic() - start_time
                    logger.info("Processed %d frames (%.1fs elapsed)", frame_num, elapsed)
                else:
                    logger.info("Processed %d frames", frame_num)

        logger.info("Processing complete: %d frames", frame_num)

        result = metrics.finalize(
            video_path=video_path,
            total_frames=frame_num,
        )
        self.reporter.generate(result)
        logger.info("Report saved to %s", self.config.output_dir)
```

- [ ] **Step 20: Run the new pipeline test**

Run: `pytest tests/test_pipeline.py::test_pipeline_uses_video_source -v`
Expected: PASS

- [ ] **Step 21: Run all existing tests (except integration)**

Run: `pytest tests/ -v --ignore=tests/test_pipeline.py -k "not integration"`
Expected: All existing tests PASS (no regressions)

- [ ] **Step 22: Commit**

```bash
git add traffic_detection_kpi/pipeline.py tests/test_pipeline.py
git commit -m "refactor: pipeline accepts VideoSource, adds graceful shutdown for live streams"
```

---

## Chunk 4: Graceful Shutdown Tests

### Task 9: Test SIGINT graceful shutdown

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] **Step 23: Write shutdown test**

Add to `tests/test_pipeline.py`:

```python
import signal
import os


def test_pipeline_graceful_shutdown_on_sigint(tmp_path):
    """Live source pipeline stops on SIGINT and still generates report."""
    from traffic_detection_kpi.config import load_config
    from traffic_detection_kpi.pipeline import VideoPipeline

    config_content = f"""\
video_path: "dummy.mp4"
output_dir: "{tmp_path}"
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
    config = load_config(str(config_path))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    call_count = 0

    def mock_read():
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            # Simulate Ctrl+C on third read
            os.kill(os.getpid(), signal.SIGINT)
        return True, frame

    mock_source = MagicMock()
    mock_source.fps = 30
    mock_source.is_live = True
    mock_source.read.side_effect = mock_read

    mock_detector = MagicMock()
    mock_detector.detect.return_value = []

    mock_tracker = MagicMock()
    mock_tracker.track.return_value = []

    pipeline = VideoPipeline(config, source=mock_source)
    pipeline.detector = mock_detector
    pipeline.tracker = mock_tracker

    original_handler = signal.getsignal(signal.SIGINT)
    pipeline.run()

    # Pipeline should have processed frames before shutdown
    assert call_count >= 3
    mock_source.release.assert_called_once()
    # Verify SIGINT handler was restored to pre-run state
    assert signal.getsignal(signal.SIGINT) is original_handler
```

- [ ] **Step 24: Run shutdown test**

Run: `pytest tests/test_pipeline.py::test_pipeline_graceful_shutdown_on_sigint -v`
Expected: PASS

- [ ] **Step 25: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: add graceful shutdown test for live source pipeline"
```

---

## Chunk 5: Config and CLI Changes

### Task 10: Make video_path optional in config

**Files:**
- Modify: `traffic_detection_kpi/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 26: Write test for config without video_path**

Add to `tests/test_config.py`:

```python
def test_load_config_without_video_path(tmp_path):
    """Config loads successfully when video_path is absent."""
    from traffic_detection_kpi.config import load_config

    config_content = f"""\
output_dir: "{tmp_path}"
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
    config = load_config(str(config_path))
    assert config.video_path is None
```

- [ ] **Step 27: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_load_config_without_video_path -v`
Expected: FAIL — `ValueError: Missing required field: 'video_path'`

- [ ] **Step 27b: Update existing test_missing_video_path**

In `tests/test_config.py`, replace `test_missing_video_path` (lines 49-54) with:

```python
def test_missing_video_path_returns_none(tmp_path):
    from traffic_detection_kpi.config import load_config
    yaml = VALID_YAML.replace('video_path: "test.mp4"\n', '')
    path = _write_yaml(tmp_path, yaml)
    config = load_config(path)
    assert config.video_path is None
```

- [ ] **Step 28: Make video_path optional in config.py**

In `traffic_detection_kpi/config.py`:

Change the `Config` dataclass:
```python
@dataclass
class Config:
    video_path: str | None
    output_dir: str
    model: ModelConfig
    tracker: TrackerConfig
    lanes: list[LaneConfig]
```

Change `load_config` — replace the `_validate_required` call:
```python
_validate_required(raw, ["output_dir", "model", "tracker", "lanes"])
```

Change the `Config` construction:
```python
return Config(
    video_path=raw.get("video_path"),
    output_dir=raw["output_dir"],
    model=model,
    tracker=tracker,
    lanes=lanes,
)
```

- [ ] **Step 29: Run config tests**

Run: `pytest tests/test_config.py -v`
Expected: All tests PASS

- [ ] **Step 30: Commit**

```bash
git add traffic_detection_kpi/config.py tests/test_config.py
git commit -m "feat: make video_path optional in config for live stream sources"
```

### Task 11: CLI flags and source factory

**Files:**
- Modify: `traffic_detection_kpi/__main__.py`

- [ ] **Step 31: Update __main__.py with --youtube and --rtsp flags**

Replace the full content of `__main__.py`:

```python
import argparse
import logging
import sys

from traffic_detection_kpi.config import load_config
from traffic_detection_kpi.pipeline import VideoPipeline
from traffic_detection_kpi.source import FileSource, YouTubeSource, RtspSource, VideoSource


def _build_source(args, config) -> VideoSource:
    cli_sources = []
    if args.youtube:
        cli_sources.append(("youtube", args.youtube))
    if args.rtsp:
        cli_sources.append(("rtsp", args.rtsp))

    if len(cli_sources) > 1:
        print("Error: only one of --youtube or --rtsp may be specified", file=sys.stderr)
        sys.exit(1)

    if cli_sources:
        kind, url = cli_sources[0]
        if kind == "youtube":
            return YouTubeSource(url)
        else:
            return RtspSource(url)

    if config.video_path:
        return FileSource(config.video_path)

    print("Error: no video source provided. Use --youtube, --rtsp, or set video_path in config.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Detection KPI — video analytics"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--youtube", metavar="URL", help="YouTube live stream URL"
    )
    parser.add_argument(
        "--rtsp", metavar="URL", help="RTSP or RTMP stream URL"
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
    source = _build_source(args, config)
    pipeline = VideoPipeline(config, source=source)
    pipeline.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 32: Run all tests**

Run: `pytest tests/ -v -k "not integration"`
Expected: All tests PASS

- [ ] **Step 33: Commit**

```bash
git add traffic_detection_kpi/__main__.py
git commit -m "feat: add --youtube and --rtsp CLI flags with source factory"
```

### Task 12: Add yt-dlp dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 34: Add yt-dlp to dependencies**

In `pyproject.toml`, add `"yt-dlp"` to the `dependencies` list:

```toml
dependencies = [
    "ultralytics>=8.0",
    "deep-sort-realtime>=1.3",
    "opencv-python>=4.8",
    "shapely>=2.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "pyyaml>=6.0",
    "yt-dlp",
]
```

- [ ] **Step 35: Install updated dependencies**

Run: `pip install -e ".[dev]"`

- [ ] **Step 36: Run full test suite**

Run: `pytest tests/ -v -k "not integration"`
Expected: All tests PASS

- [ ] **Step 37: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add yt-dlp dependency for YouTube stream support"
```

---

## Chunk 6: Final Verification

### Task 13: Full regression check

- [ ] **Step 38: Run complete test suite**

Run: `pytest tests/ -v -k "not integration"`
Expected: All tests PASS, no regressions

- [ ] **Step 39: Verify CLI help**

Run: `python -m traffic_detection_kpi --help`
Expected: Shows `--youtube URL`, `--rtsp URL`, `--config`, `--verbose` options

- [ ] **Step 40: Verify error messages**

Run: `python -m traffic_detection_kpi --config <some-config-without-video-path.yaml>`
Expected: Clear error about no video source provided

Run: `python -m traffic_detection_kpi --config config.yaml --youtube "url" --rtsp "url"`
Expected: Clear error about only one source flag allowed
