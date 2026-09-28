import pytest

from traffic_detection_kpi import TrackedObject, LaneMetrics
from traffic_detection_kpi.metrics import MetricsCollector


def _make_obj(track_id: int, class_name: str = "car", class_id: int = 2) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=(0, 0, 20, 20),
        class_id=class_id,
        class_name=class_name,
        center=(10, 10),
    )


def _empty(*lane_names: str) -> dict:
    return {name: [] for name in lane_names}


def test_queue_length_counts_objects_per_frame():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    mc.update({"L1": [_make_obj(1), _make_obj(2)]})
    for _ in range(29):
        mc.update({"L1": [_make_obj(1), _make_obj(2)]})
    result = mc.finalize()
    assert result.lanes["L1"].queue_length_timeseries[0] == 2


def test_throughput_counted_when_vehicle_leaves_lane():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=5)
    for _ in range(30):
        mc.update({"L1": [_make_obj(1)]})
    # Still present — passage not complete yet.
    assert mc.snapshot()["lanes"]["L1"]["throughput_total"] == 0
    for _ in range(6):
        mc.update(_empty("L1"))
    assert mc.snapshot()["lanes"]["L1"]["throughput_total"] == 1


def test_throughput_flushes_vehicles_still_present_at_finalize():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    for _ in range(30):
        mc.update({"L1": [_make_obj(1)]})
    result = mc.finalize()
    assert result.lanes["L1"].throughput_total == 1


def test_throughput_counts_vehicle_faster_than_one_second():
    """A vehicle crossing in 25 frames (0.83s at 30fps) must still be counted."""
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    for _ in range(25):
        mc.update({"L1": [_make_obj(1)]})
    result = mc.finalize()
    assert result.lanes["L1"].throughput_total == 1


def test_throughput_counts_vehicle_in_every_lane_it_crosses():
    """A lane-changer passes through all three lanes and counts in each."""
    lanes = ["L1", "L2", "L3"]
    mc = MetricsCollector(lane_names=lanes, video_fps=30, max_age=20)
    for lane in lanes:
        for _ in range(20):
            mc.update({**_empty(*lanes), lane: [_make_obj(1)]})
    result = mc.finalize()
    assert {k: v.throughput_total for k, v in result.lanes.items()} == {"L1": 1, "L2": 1, "L3": 1}


def test_throughput_ignores_momentary_lane_clipping():
    """A bbox clipping a lane for 2 frames is jitter, not a passage."""
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    for _ in range(2):
        mc.update({"L1": [_make_obj(1)]})
    result = mc.finalize()
    assert result.lanes["L1"].throughput_total == 0


def test_vehicle_not_double_counted_when_reentering_same_lane():
    """Bbox jitter across a lane boundary must not inflate the count."""
    lanes = ["L1", "L2"]
    mc = MetricsCollector(lane_names=lanes, video_fps=30, max_age=20)
    for _ in range(3):
        for _ in range(10):
            mc.update({"L1": [_make_obj(1)], "L2": []})
        for _ in range(10):
            mc.update({"L1": [], "L2": [_make_obj(1)]})
    result = mc.finalize()
    assert result.lanes["L1"].throughput_total == 1
    assert result.lanes["L2"].throughput_total == 1


def test_vehicle_class_breakdown():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    car = _make_obj(1, "car", 2)
    truck = _make_obj(2, "truck", 7)
    for _ in range(30):
        mc.update({"L1": [car, truck]})
    result = mc.finalize()
    assert result.lanes["L1"].vehicle_counts["car"] == 1
    assert result.lanes["L1"].vehicle_counts["truck"] == 1


def test_dwell_time_timeseries():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    obj = _make_obj(1)
    for _ in range(30):
        mc.update({"L1": [obj]})
    result = mc.finalize()
    assert len(result.lanes["L1"].avg_dwell_time_timeseries) == 1
    assert result.lanes["L1"].avg_dwell_time_timeseries[0] > 0


def test_dwell_time_resets_when_vehicle_changes_lane():
    """Dwell is time in the current lane, not total time tracked."""
    lanes = ["L1", "L2"]
    mc = MetricsCollector(lane_names=lanes, video_fps=30, max_age=20)
    for _ in range(60):
        mc.update({"L1": [_make_obj(1)], "L2": []})
    for _ in range(3):
        mc.update({"L1": [], "L2": [_make_obj(1)]})
    snap = mc.snapshot()
    assert snap["lanes"]["L2"]["avg_dwell"] == pytest.approx(3 / 30)


def test_track_pruning():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=5)
    obj = _make_obj(1)
    for _ in range(3):
        mc.update({"L1": [obj]})
    for _ in range(6):
        mc.update({"L1": []})
    assert 1 not in mc._dwell_frames


def test_multiple_lanes():
    mc = MetricsCollector(lane_names=["L1", "L2"], video_fps=30, max_age=20)
    obj1 = _make_obj(1)
    obj2 = _make_obj(2)
    for _ in range(30):
        mc.update({"L1": [obj1], "L2": [obj2]})
    result = mc.finalize()
    assert result.lanes["L1"].throughput_total == 1
    assert result.lanes["L2"].throughput_total == 1


def test_update_with_zero_fps_does_not_crash():
    mc = MetricsCollector(lane_names=["L1"], video_fps=0, max_age=20)
    mc.update({"L1": [_make_obj(1)]})
    result = mc.finalize()
    assert result.lanes["L1"].queue_length_timeseries == []
    assert result.duration_seconds == 0.0


def test_finalize_uses_explicit_duration_for_rate():
    """Live sources pass wall-clock elapsed instead of frames/fps."""
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    for _ in range(30):
        mc.update({"L1": [_make_obj(1)]})
    result = mc.finalize(duration_seconds=10.0)
    assert result.duration_seconds == 10.0
    assert result.lanes["L1"].throughput_rate_avg == pytest.approx(0.1)


def test_snapshot_uses_explicit_elapsed_for_rate():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=5)
    for _ in range(30):
        mc.update({"L1": [_make_obj(1)]})
    for _ in range(6):
        mc.update({"L1": []})
    snap = mc.snapshot(elapsed_seconds=10.0)
    assert snap["lanes"]["L1"]["throughput_rate"] == pytest.approx(0.1)


def test_snapshot_returns_current_state():
    mc = MetricsCollector(lane_names=["L1", "L2"], video_fps=30, max_age=20)
    car1 = _make_obj(1, "car", 2)
    bus2 = _make_obj(2, "bus", 5)

    for _ in range(30):
        mc.update({"L1": [car1], "L2": [bus2]})

    snap = mc.snapshot()

    assert "lanes" in snap
    assert "elapsed_frames" in snap
    assert snap["elapsed_frames"] == 30

    l1 = snap["lanes"]["L1"]
    assert l1["queue_length"] == 1
    assert l1["avg_dwell"] > 0

    l2 = snap["lanes"]["L2"]
    assert l2["queue_length"] == 1


def test_snapshot_empty_lanes():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    mc.update({"L1": []})

    snap = mc.snapshot()
    l1 = snap["lanes"]["L1"]
    assert l1["queue_length"] == 0
    assert l1["throughput_total"] == 0
    assert l1["avg_dwell"] == 0.0
    assert l1["vehicle_counts"] == {}
