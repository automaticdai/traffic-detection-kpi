from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

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
