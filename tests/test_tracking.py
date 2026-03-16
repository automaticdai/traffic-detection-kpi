from unittest.mock import MagicMock, patch
from traffic_detection_kpi import Detection, TrackedObject


def test_track_returns_tracked_objects():
    from traffic_detection_kpi.tracking import DeepSortTracker
    from traffic_detection_kpi.config import TrackerConfig

    config = TrackerConfig(
        type="deepsort", max_age=20, n_init=2,
        max_cosine_distance=0.8, embedder="mobilenet",
    )

    mock_track = MagicMock()
    mock_track.is_confirmed.return_value = True
    mock_track.track_id = 42
    mock_track.to_ltrb.return_value = [100, 200, 300, 400]
    mock_track.get_det_class.return_value = "car"

    with patch("traffic_detection_kpi.tracking.DeepSort") as MockDS:
        mock_ds = MagicMock()
        mock_ds.update_tracks.return_value = [mock_track]
        MockDS.return_value = mock_ds

        tracker = DeepSortTracker(config)
        detections = [Detection(bbox=(100, 200, 200, 200), class_id=2, class_name="car", confidence=0.9)]
        result = tracker.track(detections, MagicMock())

    assert len(result) == 1
    assert isinstance(result[0], TrackedObject)
    assert result[0].track_id == 42
    assert result[0].bbox == (100, 200, 300, 400)
    assert result[0].center == (200, 300)  # midpoint of ltrb
    assert result[0].class_name == "car"
    assert result[0].class_id == 2


def test_track_skips_unconfirmed():
    from traffic_detection_kpi.tracking import DeepSortTracker
    from traffic_detection_kpi.config import TrackerConfig

    config = TrackerConfig(
        type="deepsort", max_age=20, n_init=2,
        max_cosine_distance=0.8, embedder="mobilenet",
    )

    mock_track = MagicMock()
    mock_track.is_confirmed.return_value = False

    with patch("traffic_detection_kpi.tracking.DeepSort") as MockDS:
        mock_ds = MagicMock()
        mock_ds.update_tracks.return_value = [mock_track]
        MockDS.return_value = mock_ds

        tracker = DeepSortTracker(config)
        result = tracker.track([], MagicMock())

    assert len(result) == 0
