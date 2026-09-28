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


def test_find_nearest_vertex_within_threshold():
    from traffic_detection_kpi.lane_editor import find_nearest_vertex

    polygon = [[100, 100], [200, 100], [200, 200], [100, 200]]
    result = find_nearest_vertex((100, 100), polygon, threshold=15)
    assert result == 0

    result = find_nearest_vertex((210, 100), polygon, threshold=15)
    assert result == 1

    result = find_nearest_vertex((150, 150), polygon, threshold=15)
    assert result is None


def test_find_nearest_edge_within_threshold():
    from traffic_detection_kpi.lane_editor import find_nearest_edge

    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    result = find_nearest_edge((50, 5), polygon, threshold=10)
    assert result == 0

    result = find_nearest_edge((50, 97), polygon, threshold=10)
    assert result == 2

    result = find_nearest_edge((50, 50), polygon, threshold=10)
    assert result is None


def test_project_point_on_edge_returns_int():
    from traffic_detection_kpi.lane_editor import project_point_on_edge

    result = project_point_on_edge((50, 5), [0, 0], [100, 0])
    assert result == [50, 0]
    assert isinstance(result[0], int)
    assert isinstance(result[1], int)

    result = project_point_on_edge((30, 25), [0, 0], [100, 100])
    assert isinstance(result[0], int)
    assert isinstance(result[1], int)


def test_no_delete_below_three_vertices():
    from traffic_detection_kpi.lane_editor import can_delete_vertex

    assert can_delete_vertex(4) is True
    assert can_delete_vertex(3) is False
    assert can_delete_vertex(2) is False


def _editor(monkeypatch, frame_w, frame_h, polygon):
    import numpy as np
    from traffic_detection_kpi.lane_editor import LaneEditor

    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    editor = LaneEditor(frame, [{"name": "L1", "polygon": polygon}], [(0, 200, 0)])
    monkeypatch.setattr(editor, "_redraw", lambda: None)
    return editor


def test_vertex_hit_threshold_scales_with_display(monkeypatch):
    """On a downscaled frame the grab radius must stay constant on screen."""
    import cv2

    # 2560x1440 displays at scale 0.5, so a vertex at frame (200,200) sits at (100,100).
    editor = _editor(monkeypatch, 2560, 1440, [[200, 200], [800, 200], [800, 800], [200, 800]])
    assert editor._scale == 0.5

    # 10px away on screen — comfortably inside a 15px on-screen grab radius.
    editor._mouse_cb(cv2.EVENT_LBUTTONDOWN, 110, 100, 0, None)
    assert editor._sel_vert_idx == 0


def test_vertex_hit_threshold_unchanged_at_full_scale(monkeypatch):
    import cv2

    editor = _editor(monkeypatch, 640, 480, [[200, 200], [400, 200], [400, 400], [200, 400]])
    assert editor._scale == 1.0

    editor._mouse_cb(cv2.EVENT_LBUTTONDOWN, 210, 200, 0, None)
    assert editor._sel_vert_idx == 0

    editor._sel_vert_idx = None
    editor._mouse_cb(cv2.EVENT_LBUTTONDOWN, 300, 300, 0, None)
    assert editor._sel_vert_idx is None
