import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestFileSource:
    def test_read_returns_frame(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = FileSource("video.mp4")
            ret, result = source.read()

        assert ret is True
        assert result is not None
        assert result.shape == (480, 640, 3)

    def test_fps_from_capture(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = FileSource("video.mp4")

        assert source.fps == 25

    def test_is_live_false(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = FileSource("video.mp4")

        assert source.is_live is False

    def test_release_delegates_to_capture(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = FileSource("video.mp4")
            source.release()

        mock_cap.release.assert_called_once()

    def test_raises_on_unopenable_file(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(RuntimeError, match="Cannot open video"):
                FileSource("nonexistent.mp4")

    def test_fps_rounds_not_truncates(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 29.97

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = FileSource("video.mp4")

        assert source.fps == 30  # round(29.97), not int(29.97)=29

    def test_read_eof(self):
        from traffic_detection_kpi.source import FileSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (False, None)

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = FileSource("video.mp4")
            ret, frame = source.read()

        assert ret is False
        assert frame is None
