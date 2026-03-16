from traffic_detection_kpi import TrackedObject
from traffic_detection_kpi.lanes import LaneZone


def _make_obj(track_id: int, center: tuple[int, int]) -> TrackedObject:
    cx, cy = center
    return TrackedObject(
        track_id=track_id,
        bbox=(cx - 10, cy - 10, cx + 10, cy + 10),
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


def test_classify_first_match_wins():
    lane1 = LaneZone("First", [[0, 0], [100, 0], [100, 100], [0, 100]])
    lane2 = LaneZone("Second", [[25, 25], [75, 25], [75, 75], [25, 75]])
    obj = _make_obj(1, (50, 50))
    result = LaneZone.classify([lane1, lane2], [obj])
    assert len(result["First"]) == 1
    assert len(result["Second"]) == 0


def test_classify_excludes_objects_outside_all_lanes():
    lane = LaneZone("L1", [[0, 0], [50, 0], [50, 50], [0, 50]])
    obj = _make_obj(1, (200, 200))
    result = LaneZone.classify([lane], [obj])
    assert len(result["L1"]) == 0
