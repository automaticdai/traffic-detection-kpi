from traffic_detection_kpi import TrackedObject
from traffic_detection_kpi.lanes import LaneZone


def _make_obj(track_id: int, center: tuple[int, int], half_w: int = 10, half_h: int = 10) -> TrackedObject:
    cx, cy = center
    return TrackedObject(
        track_id=track_id,
        bbox=(cx - half_w, cy - half_h, cx + half_w, cy + half_h),
        class_id=2,
        class_name="car",
        center=center,
    )


def test_contains_point_inside():
    lane = LaneZone("L1", [[0, 0], [100, 0], [100, 100], [0, 100]])
    assert lane.contains((50, 50)) is True


def test_contains_point_outside():
    lane = LaneZone("L1", [[0, 0], [100, 0], [100, 100], [0, 100]])
    assert lane.contains((200, 200)) is False


def test_overlap_ratio_fully_inside():
    lane = LaneZone("L1", [[0, 0], [200, 0], [200, 200], [0, 200]])
    ratio = lane.overlap_ratio((50, 50, 100, 100))
    assert ratio > 0.99


def test_overlap_ratio_no_overlap():
    lane = LaneZone("L1", [[0, 0], [50, 0], [50, 50], [0, 50]])
    ratio = lane.overlap_ratio((100, 100, 150, 150))
    assert ratio == 0.0


def test_overlap_ratio_partial():
    # Lane covers left half (0-50), bbox spans 25-75 -> 50% overlap
    lane = LaneZone("L1", [[0, 0], [50, 0], [50, 100], [0, 100]])
    ratio = lane.overlap_ratio((25, 25, 75, 75))
    assert 0.45 < ratio < 0.55


def test_classify_assigns_to_correct_lane():
    lane1 = LaneZone("Left", [[0, 0], [50, 0], [50, 100], [0, 100]])
    lane2 = LaneZone("Right", [[50, 0], [100, 0], [100, 100], [50, 100]])
    obj_left = _make_obj(1, (25, 50))
    obj_right = _make_obj(2, (75, 50))
    result = LaneZone.classify([lane1, lane2], [obj_left, obj_right])
    assert len(result["Left"]) == 1
    assert result["Left"][0].track_id == 1
    assert len(result["Right"]) == 1
    assert result["Right"][0].track_id == 2


def test_classify_highest_overlap_wins():
    """When a car overlaps two lanes, it goes to the one with more overlap."""
    # Lane1: 0-60, Lane2: 40-100. Bbox: 20-70 (50px wide)
    # Overlap with Lane1: 40px/50px = 80%, Lane2: 20px/50px = 40%
    lane1 = LaneZone("First", [[0, 0], [60, 0], [60, 100], [0, 100]])
    lane2 = LaneZone("Second", [[40, 0], [100, 0], [100, 100], [40, 100]])
    obj = _make_obj(1, (45, 50), half_w=25, half_h=25)  # bbox: 20-70
    result = LaneZone.classify([lane1, lane2], [obj])
    assert len(result["First"]) == 1
    assert len(result["Second"]) == 0


def test_classify_partial_overlap_counted():
    """A car partially in a lane (>25% overlap) should be counted."""
    lane = LaneZone("L1", [[0, 0], [50, 0], [50, 100], [0, 100]])
    # Bbox 30-70: overlap is 20px out of 40px = 50% -> above 25% threshold
    obj = _make_obj(1, (50, 50), half_w=20, half_h=20)
    result = LaneZone.classify([lane], [obj])
    assert len(result["L1"]) == 1


def test_classify_tiny_overlap_excluded():
    """A car barely touching a lane (<25% overlap) should NOT be counted."""
    lane = LaneZone("L1", [[0, 0], [50, 0], [50, 100], [0, 100]])
    # Bbox 42-82: overlap is 8px out of 40px = 20% -> below 25% threshold
    obj = _make_obj(1, (62, 50), half_w=20, half_h=20)
    result = LaneZone.classify([lane], [obj])
    assert len(result["L1"]) == 0


def test_classify_excludes_objects_outside_all_lanes():
    lane = LaneZone("L1", [[0, 0], [50, 0], [50, 50], [0, 50]])
    obj = _make_obj(1, (200, 200))
    result = LaneZone.classify([lane], [obj])
    assert len(result["L1"]) == 0


def test_classify_each_car_binds_to_one_lane():
    """Even if a car overlaps multiple lanes above threshold, it only appears in one."""
    lane1 = LaneZone("A", [[0, 0], [60, 0], [60, 100], [0, 100]])
    lane2 = LaneZone("B", [[30, 0], [100, 0], [100, 100], [30, 100]])
    # Bbox 20-60: overlaps both lanes above 25%
    obj = _make_obj(1, (40, 50), half_w=20, half_h=20)
    result = LaneZone.classify([lane1, lane2], [obj])
    total = len(result["A"]) + len(result["B"])
    assert total == 1
