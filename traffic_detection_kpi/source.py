from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

import yt_dlp

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class VideoSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame."""
        ...

    @property
    def fps(self) -> int:
        """Frames per second (rounded to nearest integer)."""
        ...

    @property
    def is_live(self) -> bool:
        """True for live streams, False for recorded files."""
        ...

    @property
    def url(self) -> str:
        """The source path or URL (for provenance in reports)."""
        ...

    def release(self) -> None:
        """Release underlying resources."""
        ...


class FileSource:
    def __init__(self, path: str) -> None:
        self._url = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self._fps = round(self._cap.get(cv2.CAP_PROP_FPS)) or 30

    def read(self) -> tuple[bool, np.ndarray | None]:
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def is_live(self) -> bool:
        return False

    @property
    def url(self) -> str:
        return self._url

    def release(self) -> None:
        self._cap.release()


class YouTubeSource:
    _MAX_RETRIES = 5
    _RETRY_DELAY = 1.0

    def __init__(self, url: str) -> None:
        self._url = url
        stream_url, meta_fps = self._resolve(url)
        self._cap = cv2.VideoCapture(stream_url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open resolved YouTube stream: {url}")
        self._fps = self._determine_fps(meta_fps)

    @staticmethod
    def _resolve(url: str) -> tuple[str, float | None]:
        opts = {
            "format": "best[height<=720][ext=mp4]/best[height<=720]/best",
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            raise RuntimeError(f"Failed to resolve YouTube URL: {e}") from e
        return info["url"], info.get("fps")

    def _determine_fps(self, meta_fps: float | None) -> int:
        if meta_fps:
            return round(meta_fps)
        cv_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if cv_fps > 0:
            return round(cv_fps)
        return 30

    def read(self) -> tuple[bool, np.ndarray | None]:
        ret, frame = self._cap.read()
        if ret:
            return True, frame
        for attempt in range(self._MAX_RETRIES):
            logger.warning("YouTube read failed, retry %d/%d", attempt + 1, self._MAX_RETRIES)
            time.sleep(self._RETRY_DELAY)
            ret, frame = self._cap.read()
            if ret:
                return True, frame
        return False, None

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def is_live(self) -> bool:
        return True

    @property
    def url(self) -> str:
        return self._url

    def release(self) -> None:
        self._cap.release()


class RtspSource:
    _MAX_RETRIES = 5
    _RETRY_DELAY = 1.0

    def __init__(self, url: str) -> None:
        self._url = url
        self._cap = cv2.VideoCapture(url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {url}")
        cv_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = round(cv_fps) if cv_fps > 0 else 30

    def read(self) -> tuple[bool, np.ndarray | None]:
        ret, frame = self._cap.read()
        if ret:
            return True, frame
        for attempt in range(self._MAX_RETRIES):
            logger.warning("Stream read failed, retry %d/%d", attempt + 1, self._MAX_RETRIES)
            time.sleep(self._RETRY_DELAY)
            ret, frame = self._cap.read()
            if ret:
                return True, frame
        return False, None

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def is_live(self) -> bool:
        return True

    @property
    def url(self) -> str:
        return self._url

    def release(self) -> None:
        self._cap.release()
