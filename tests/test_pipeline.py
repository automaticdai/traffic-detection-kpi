"""Pipeline tests — unit tests + integration test.

Integration test requires a video file and YOLO model to run.
Skip by default; run with: pytest tests/test_pipeline.py -v --run-integration
"""
import json
import signal
import os

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


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
    mock_source.url = "dummy.mp4"
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
    mock_source.url = "https://youtube.com/test"
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


@pytest.mark.integration
def test_full_pipeline(tmp_path):
    """Run the full pipeline on a short video and verify output structure."""
    from traffic_detection_kpi.config import load_config
    from traffic_detection_kpi.pipeline import VideoPipeline

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

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "charts" / "throughput_by_lane.png").exists()
    assert (tmp_path / "charts" / "queue_length_over_time.png").exists()
    assert (tmp_path / "charts" / "dwell_time_over_time.png").exists()
    assert (tmp_path / "charts" / "vehicle_class_breakdown.png").exists()

    data = json.loads((tmp_path / "metrics.json").read_text())
    assert "lanes" in data
    assert "Lane 1" in data["lanes"]
    assert "throughput_total" in data["lanes"]["Lane 1"]
    assert "queue_length_timeseries" in data["lanes"]["Lane 1"]


def _write_config(tmp_path, lanes='[[0, 0], [100, 0], [100, 100], [0, 100]]'):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""\
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
    polygon: {lanes}
""")
    return config_path


def _mock_source(is_live=False, frames=2, fps=30):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    source = MagicMock()
    source.fps = fps
    source.is_live = is_live
    source.url = "dummy.mp4"
    source.read.side_effect = [(True, frame)] * frames + [(False, None)]
    return source


def _stub_pipeline(config, source, show=False):
    from traffic_detection_kpi.pipeline import VideoPipeline

    pipeline = VideoPipeline(config, source=source, show=show)
    pipeline.detector = MagicMock(**{"detect.return_value": []})
    pipeline.tracker = MagicMock(**{"track.return_value": []})
    return pipeline


def test_show_skipped_when_no_display_available(tmp_path, monkeypatch):
    """--show on a headless box must degrade, not abort the process."""
    from traffic_detection_kpi.config import load_config
    from traffic_detection_kpi import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    config = load_config(str(_write_config(tmp_path)))
    source = _mock_source()

    with patch.object(pipeline_mod.cv2, "namedWindow") as mock_window:
        _stub_pipeline(config, source, show=True).run()

    mock_window.assert_not_called()
    source.release.assert_called_once()
    assert (tmp_path / "metrics.json").exists()


def test_source_released_when_window_creation_fails(tmp_path, monkeypatch):
    """A GUI failure must not leak the capture handle."""
    import cv2

    from traffic_detection_kpi.config import load_config
    from traffic_detection_kpi import pipeline as pipeline_mod

    monkeypatch.setenv("DISPLAY", ":0")
    config = load_config(str(_write_config(tmp_path)))
    source = _mock_source()

    with patch.object(pipeline_mod.cv2, "namedWindow", side_effect=cv2.error("no gui")):
        _stub_pipeline(config, source, show=True).run()

    source.release.assert_called_once()


def test_live_source_reports_wall_clock_duration(tmp_path):
    """Frames/fps is fiction on a live stream that drops frames."""
    from traffic_detection_kpi.config import load_config
    from traffic_detection_kpi import pipeline as pipeline_mod

    config = load_config(str(_write_config(tmp_path)))
    source = _mock_source(is_live=True, frames=2)

    with patch.object(pipeline_mod.time, "monotonic", side_effect=[100.0, 110.0]):
        _stub_pipeline(config, source).run()

    data = json.loads((tmp_path / "metrics.json").read_text())
    assert data["duration_seconds"] == 10.0


def test_file_source_reports_frame_derived_duration(tmp_path):
    from traffic_detection_kpi.config import load_config

    config = load_config(str(_write_config(tmp_path)))
    _stub_pipeline(config, _mock_source(is_live=False, frames=2, fps=30)).run()

    data = json.loads((tmp_path / "metrics.json").read_text())
    assert data["duration_seconds"] == pytest.approx(2 / 30)


def test_detector_receives_class_ids_from_config(tmp_path):
    """class_ids must reach YOLO so filtering happens during inference."""
    from traffic_detection_kpi.config import load_config
    from traffic_detection_kpi.pipeline import VideoPipeline

    config = load_config(str(_write_config(tmp_path)))
    with patch("traffic_detection_kpi.pipeline.YoloDetector") as MockDetector:
        VideoPipeline(config, source=_mock_source())

    assert MockDetector.call_args.kwargs["class_ids"] == [2]
