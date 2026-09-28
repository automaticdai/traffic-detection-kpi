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


class TestRtspSource:
    def test_opens_rtsp_url(self):
        from traffic_detection_kpi.source import RtspSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = RtspSource("rtsp://camera.example.com/stream")
            ret, result = source.read()

        assert ret is True
        assert result is not None
        assert source.is_live is True
        assert source.fps == 25

    def test_fps_fallback_to_30(self):
        from traffic_detection_kpi.source import RtspSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0.0

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = RtspSource("rtsp://camera.example.com/stream")

        assert source.fps == 30

    def test_raises_on_unreachable(self):
        from traffic_detection_kpi.source import RtspSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(RuntimeError, match="Cannot open stream"):
                RtspSource("rtsp://bad.example.com/stream")

    def test_rtmp_url(self):
        from traffic_detection_kpi.source import RtspSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap):
            source = RtspSource("rtmp://camera.example.com/live/stream")

        assert source.is_live is True

    def test_retry_on_transient_failure(self):
        from traffic_detection_kpi.source import RtspSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(False, None), (True, frame)]

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap), \
             patch("traffic_detection_kpi.source.time.sleep"):
            source = RtspSource("rtsp://camera.example.com/stream")
            ret, result = source.read()

        assert ret is True

    def test_retry_exhausted(self):
        from traffic_detection_kpi.source import RtspSource

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (False, None)

        with patch("traffic_detection_kpi.source.cv2.VideoCapture", return_value=mock_cap), \
             patch("traffic_detection_kpi.source.time.sleep"):
            source = RtspSource("rtsp://camera.example.com/stream")
            ret, frame = source.read()

        assert ret is False
        assert mock_cap.read.call_count == 6  # 1 initial + 5 retries


class TestYouTubeFormatSelection:
    """YouTube serves live streams as HLS with no muxed (video+audio) format.

    yt-dlp's `best` selector only matches formats carrying both streams, so it
    matches nothing on a live stream and resolution fails outright.
    """

    _LIVE_FORMATS = [
        {"format_id": "233", "url": "u", "ext": "mp4", "vcodec": "none",
         "acodec": "mp4a.40.5", "protocol": "m3u8_native"},
        {"format_id": "230", "url": "u", "ext": "mp4", "vcodec": "avc1.4D401E",
         "acodec": "none", "height": 360, "fps": 30, "protocol": "m3u8_native"},
        {"format_id": "232", "url": "u", "ext": "mp4", "vcodec": "avc1.4D401F",
         "acodec": "none", "height": 720, "fps": 30, "protocol": "m3u8_native"},
    ]

    def _select(self, format_string, formats):
        import yt_dlp

        ydl = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "simulate": True})
        selector = ydl.build_format_selector(format_string)
        return [f["format_id"] for f in selector(
            {"formats": formats, "incomplete_formats": False}
        )]

    def test_selects_video_only_format_from_live_stream(self):
        from traffic_detection_kpi.source import YDL_FORMAT

        assert self._select(YDL_FORMAT, self._LIVE_FORMATS) == ["232"]

    def test_prefers_720p_when_higher_resolutions_offered(self):
        from traffic_detection_kpi.source import YDL_FORMAT

        formats = self._LIVE_FORMATS + [
            {"format_id": "270", "url": "u", "ext": "mp4", "vcodec": "avc1.640028",
             "acodec": "none", "height": 1080, "fps": 30, "protocol": "m3u8_native"},
        ]
        assert self._select(YDL_FORMAT, formats) == ["232"]

    def test_still_selects_muxed_format_for_recorded_video(self):
        from traffic_detection_kpi.source import YDL_FORMAT

        formats = [
            {"format_id": "18", "url": "u", "ext": "mp4", "vcodec": "avc1.42001E",
             "acodec": "mp4a.40.2", "height": 360, "fps": 30, "protocol": "https"},
        ]
        assert self._select(YDL_FORMAT, formats) == ["18"]
