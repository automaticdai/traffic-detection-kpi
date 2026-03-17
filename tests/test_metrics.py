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


def test_queue_length_counts_objects_per_frame():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    mc.update({"L1": [_make_obj(1), _make_obj(2)]})
    for _ in range(29):
        mc.update({"L1": [_make_obj(1), _make_obj(2)]})
    result = mc.finalize()
    assert result.lanes["L1"].queue_length_timeseries[0] == 2


def test_throughput_after_one_second():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    for _ in range(30):
        mc.update({"L1": [_make_obj(1)]})
    result = mc.finalize()
    assert result.lanes["L1"].throughput_total == 1


def test_throughput_not_counted_before_threshold():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    for _ in range(10):
        mc.update({"L1": [_make_obj(1)]})
    result = mc.finalize()
    assert result.lanes["L1"].throughput_total == 0


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


def test_snapshot_returns_current_state():
    mc = MetricsCollector(lane_names=["L1", "L2"], video_fps=30, max_age=20)
    car1 = _make_obj(1, "car", 2)
    bus2 = _make_obj(2, "bus", 5)

    # Run 30 frames so throughput triggers for both
    for _ in range(30):
        mc.update({"L1": [car1], "L2": [bus2]})

    snap = mc.snapshot()

    # Structure
    assert "lanes" in snap
    assert "elapsed_frames" in snap
    assert snap["elapsed_frames"] == 30

    # L1
    l1 = snap["lanes"]["L1"]
    assert l1["queue_length"] == 1
    assert l1["throughput_total"] == 1
    assert l1["throughput_rate"] > 0
    assert l1["avg_dwell"] > 0
    assert l1["vehicle_counts"]["car"] == 1

    # L2
    l2 = snap["lanes"]["L2"]
    assert l2["queue_length"] == 1
    assert l2["throughput_total"] == 1
    assert l2["vehicle_counts"]["bus"] == 1


def test_snapshot_empty_lanes():
    mc = MetricsCollector(lane_names=["L1"], video_fps=30, max_age=20)
    mc.update({"L1": []})

    snap = mc.snapshot()
    l1 = snap["lanes"]["L1"]
    assert l1["queue_length"] == 0
    assert l1["throughput_total"] == 0
    assert l1["avg_dwell"] == 0.0
    assert l1["vehicle_counts"] == {}
