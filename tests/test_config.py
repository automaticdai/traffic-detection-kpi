import pytest
import tempfile
import os
from pathlib import Path


def _write_yaml(tmp_path: Path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return str(p)


VALID_YAML = """\
video_path: "test.mp4"
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
    polygon: [[0, 0], [100, 0], [100, 100], [0, 100]]
"""


def test_load_valid_config(tmp_path):
    from traffic_detection_kpi.config import load_config
    path = _write_yaml(tmp_path, VALID_YAML)
    config = load_config(path)
    assert config.video_path == "test.mp4"
    assert config.model.confidence == 0.2
    assert len(config.lanes) == 1
    assert config.lanes[0].name == "Lane 1"


def test_class_name_to_coco_id(tmp_path):
    from traffic_detection_kpi.config import load_config
    path = _write_yaml(tmp_path, VALID_YAML)
    config = load_config(path)
    assert config.model.class_ids == [2, 3, 5, 7]


def test_missing_video_path_returns_none(tmp_path):
    from traffic_detection_kpi.config import load_config
    yaml = VALID_YAML.replace('video_path: "test.mp4"\n', '')
    path = _write_yaml(tmp_path, yaml)
    config = load_config(path)
    assert config.video_path is None


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


def test_invalid_class_name(tmp_path):
    from traffic_detection_kpi.config import load_config
    yaml = VALID_YAML.replace("classes: [car, motorcycle, bus, truck]",
                               "classes: [car, airplane]")
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="airplane"):
        load_config(path)


def test_polygon_needs_at_least_3_points(tmp_path):
    from traffic_detection_kpi.config import load_config
    yaml = VALID_YAML.replace(
        "polygon: [[0, 0], [100, 0], [100, 100], [0, 100]]",
        "polygon: [[0, 0], [100, 0]]"
    )
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="polygon"):
        load_config(path)


def test_self_intersecting_polygon_rejected(tmp_path):
    from traffic_detection_kpi.config import load_config
    # Bowtie shape — self-intersecting
    yaml = VALID_YAML.replace(
        "polygon: [[0, 0], [100, 0], [100, 100], [0, 100]]",
        "polygon: [[0, 0], [100, 100], [100, 0], [0, 100]]"
    )
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="self-intersecting"):
        load_config(path)
