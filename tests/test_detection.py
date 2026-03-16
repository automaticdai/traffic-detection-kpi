from unittest.mock import MagicMock, patch
from traffic_detection_kpi import Detection


def test_detect_filters_by_allowed_classes():
    from traffic_detection_kpi.detection import YoloDetector

    mock_box1 = MagicMock()
    mock_box1.xyxy = [MagicMock()]
    mock_box1.xyxy[0].__iter__ = lambda s: iter([100, 200, 150, 250])
    mock_box1.cls = [MagicMock(item=lambda: 2)]  # car
    mock_box1.conf = [MagicMock(item=lambda: 0.9)]

    mock_box2 = MagicMock()
    mock_box2.xyxy = [MagicMock()]
    mock_box2.xyxy[0].__iter__ = lambda s: iter([100, 200, 150, 250])
    mock_box2.cls = [MagicMock(item=lambda: 0)]  # person — filtered out
    mock_box2.conf = [MagicMock(item=lambda: 0.8)]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box1, mock_box2]
    mock_result.names = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    with patch("traffic_detection_kpi.detection.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        MockYOLO.return_value = mock_model

        detector = YoloDetector(
            model_path="fake.pt",
            confidence=0.2,
            class_filter=["car", "truck"],
        )
        frame = MagicMock()
        detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].class_name == "car"


def test_detect_returns_detection_dataclass():
    from traffic_detection_kpi.detection import YoloDetector

    mock_box = MagicMock()
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].__iter__ = lambda s: iter([100, 200, 160, 270])
    mock_box.cls = [MagicMock(item=lambda: 2)]
    mock_box.conf = [MagicMock(item=lambda: 0.85)]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {2: "car"}

    with patch("traffic_detection_kpi.detection.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        MockYOLO.return_value = mock_model

        detector = YoloDetector("fake.pt", 0.2, ["car"])
        detections = detector.detect(MagicMock())

    assert isinstance(detections[0], Detection)
    assert detections[0].bbox == (100, 200, 60, 70)  # converted to xywh
