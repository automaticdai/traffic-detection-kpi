import numpy as np
import pytest


def _make_snapshot(lane_names):
    """Create a minimal metrics snapshot."""
    return {
        "lanes": {
            name: {
                "queue_length": 0,
                "throughput_total": 0,
                "throughput_rate": 0.0,
                "avg_dwell": 0.0,
                "vehicle_counts": {},
            }
            for name in lane_names
        },
        "elapsed_frames": 0,
    }


class TestFrameAnnotator:
    def test_draw_returns_frame_with_panel(self):
        from traffic_detection_kpi.annotator import FrameAnnotator

        lane_names = ["Lane 1"]
        lane_polygons = [[[100, 100], [200, 100], [200, 300], [100, 300]]]
        annotator = FrameAnnotator(lane_names, lane_polygons, fps=30)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        snapshot = _make_snapshot(lane_names)

        result = annotator.draw(frame, [], {}, snapshot)

        assert result.shape == (480, 940, 3)  # 640 + 300 panel
        assert result.dtype == np.uint8

    def test_draw_with_empty_inputs(self):
        from traffic_detection_kpi.annotator import FrameAnnotator

        lane_names = ["Lane 1", "Lane 2"]
        lane_polygons = [
            [[0, 0], [100, 0], [100, 100]],
            [[200, 0], [300, 0], [300, 100]],
        ]
        annotator = FrameAnnotator(lane_names, lane_polygons, fps=30)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        snapshot = _make_snapshot(lane_names)

        # Should not crash with no objects
        result = annotator.draw(frame, [], {"Lane 1": [], "Lane 2": []}, snapshot)
        assert result.shape == (480, 940, 3)

    def test_draw_with_detections(self):
        from traffic_detection_kpi.annotator import FrameAnnotator
        from traffic_detection_kpi import TrackedObject

        lane_names = ["Lane 1"]
        lane_polygons = [[[0, 0], [640, 0], [640, 480], [0, 480]]]
        annotator = FrameAnnotator(lane_names, lane_polygons, fps=30)

        obj = TrackedObject(
            track_id=1,
            bbox=(100, 100, 200, 200),
            class_id=2,
            class_name="car",
            center=(150, 150),
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        snapshot = _make_snapshot(lane_names)
        snapshot["lanes"]["Lane 1"]["queue_length"] = 1

        result = annotator.draw(frame, [obj], {"Lane 1": [obj]}, snapshot)

        assert result.shape == (480, 940, 3)
        assert result.dtype == np.uint8
        # Verify something was drawn (frame shouldn't be all zeros anymore)
        assert result[:, :640, :].sum() > 0

    def test_lane_colors_are_consistent(self):
        from traffic_detection_kpi.annotator import FrameAnnotator

        lane_names = ["A", "B", "C"]
        lane_polygons = [
            [[0, 0], [10, 0], [10, 10]],
            [[20, 0], [30, 0], [30, 10]],
            [[40, 0], [50, 0], [50, 10]],
        ]
        a1 = FrameAnnotator(lane_names, lane_polygons, fps=30)
        a2 = FrameAnnotator(lane_names, lane_polygons, fps=30)

        assert a1._lane_colors == a2._lane_colors
        assert len(a1._lane_colors) == 3
        # Each lane gets a different color
        assert a1._lane_colors[0] != a1._lane_colors[1]
