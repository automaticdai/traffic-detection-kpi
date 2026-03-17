from __future__ import annotations

import cv2
import numpy as np


class LaneEditor:
    def __init__(self, frame: np.ndarray, lanes: list[dict], lane_colors: list[tuple]) -> None:
        self._frame = frame
        self._lanes = lanes
        self._lane_colors = lane_colors

    def run(self) -> tuple[bool, list[dict]]:
        raise NotImplementedError("LaneEditor.run() not yet implemented")
