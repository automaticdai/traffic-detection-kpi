import math
import yaml
import pytest


def test_save_lanes_to_config_roundtrip(tmp_path):
    """Save modified lanes, reload, verify lanes match."""
    from traffic_detection_kpi.editor_cli import save_lanes_to_config
    from traffic_detection_kpi.config import load_config

    config_content = """\
video_path: "test.mp4"
output_dir: "./output"
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

    new_lanes = [
        {"name": "Lane A", "polygon": [[10, 10], [200, 10], [200, 200], [10, 200]]},
        {"name": "Lane B", "polygon": [[300, 300], [400, 300], [400, 400]]},
    ]
    save_lanes_to_config(str(config_path), new_lanes)

    config = load_config(str(config_path))
    assert len(config.lanes) == 2
    assert config.lanes[0].name == "Lane A"
    assert config.lanes[0].polygon == [[10, 10], [200, 10], [200, 200], [10, 200]]
    assert config.lanes[1].name == "Lane B"
    assert config.lanes[1].polygon == [[300, 300], [400, 300], [400, 400]]


def test_save_lanes_preserves_other_config(tmp_path):
    """Non-lane config fields survive save."""
    from traffic_detection_kpi.editor_cli import save_lanes_to_config

    config_content = """\
video_path: "test.mp4"
output_dir: "./output"
model:
  path: "yolo11m.pt"
  confidence: 0.2
  classes: [car, truck]
tracker:
  type: deepsort
  max_age: 20
  n_init: 2
  max_cosine_distance: 0.8
  embedder: mobilenet
lanes:
  - name: "Old"
    polygon: [[0, 0], [1, 0], [1, 1]]
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    save_lanes_to_config(str(config_path), [
        {"name": "New", "polygon": [[5, 5], [10, 5], [10, 10]]},
    ])

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    assert raw["video_path"] == "test.mp4"
    assert raw["output_dir"] == "./output"
    assert raw["model"]["path"] == "yolo11m.pt"
    assert raw["model"]["classes"] == ["car", "truck"]
    assert raw["tracker"]["type"] == "deepsort"
    assert raw["tracker"]["max_age"] == 20
    assert len(raw["lanes"]) == 1
    assert raw["lanes"][0]["name"] == "New"
