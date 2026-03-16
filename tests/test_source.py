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


class TestYouTubeSource:
    def test_resolves_url_and_reads_frame(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved-stream.example.com/video",
            "fps": 30,
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = YouTubeSource("https://www.youtube.com/watch?v=test")
            ret, result = source.read()

        assert ret is True
        assert result is not None
        assert source.is_live is True

    def test_fps_from_ytdlp_metadata_rounded(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0.0

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved.example.com/video",
            "fps": 29.97,
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = YouTubeSource("https://www.youtube.com/watch?v=test")

        assert source.fps == 30

    def test_fps_fallback_to_opencv(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved.example.com/video",
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = YouTubeSource("https://www.youtube.com/watch?v=test")

        assert source.fps == 25

    def test_fps_fallback_to_default_30(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0.0

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved.example.com/video",
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = YouTubeSource("https://www.youtube.com/watch?v=test")

        assert source.fps == 30

    def test_raises_on_invalid_url(self):
        from traffic_detection_kpi.source import YouTubeSource
        import yt_dlp

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("not found")

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl):
            with pytest.raises(RuntimeError, match="Failed to resolve YouTube URL"):
                YouTubeSource("https://www.youtube.com/watch?v=invalid")

    def test_raises_on_unopenable_stream(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved.example.com/video",
            "fps": 30,
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(RuntimeError, match="Cannot open resolved YouTube stream"):
                YouTubeSource("https://www.youtube.com/watch?v=test")

    def test_retry_on_transient_failure(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (False, None),
            (False, None),
            (True, frame),
        ]

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved.example.com/video",
            "fps": 30,
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap), \
             patch("traffic_detection_kpi.source.time.sleep"):
            source = YouTubeSource("https://www.youtube.com/watch?v=test")
            ret, result = source.read()

        assert ret is True
        assert result is not None

    def test_retry_exhausted_returns_false(self):
        from traffic_detection_kpi.source import YouTubeSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (False, None)

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "url": "https://resolved.example.com/video",
            "fps": 30,
        }

        with patch("traffic_detection_kpi.source.yt_dlp.YoutubeDL", return_value=mock_ydl), \
             patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap), \
             patch("traffic_detection_kpi.source.time.sleep"):
            source = YouTubeSource("https://www.youtube.com/watch?v=test")
            ret, frame = source.read()

        assert ret is False
        assert frame is None
        assert mock_cap.read.call_count == 6  # 1 initial + 5 retries
