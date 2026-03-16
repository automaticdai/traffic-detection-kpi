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
