from __future__ import annotations

import cv2
import numpy as np

from traffic_detection_kpi import TrackedObject

# BGR color palette for lanes
_PALETTE = [
    (0, 200, 0),      # green
    (200, 100, 0),     # blue
    (0, 140, 255),     # orange
    (0, 0, 200),       # red
    (200, 200, 0),     # cyan
    (200, 0, 200),     # magenta
    (0, 200, 200),     # yellow
    (200, 200, 200),   # light gray
]

_PANEL_WIDTH = 300
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GRAY = (128, 128, 128)


class FrameAnnotator:
    def __init__(
        self,
        lane_names: list[str],
        lane_polygons: list[list[list[int]]],
        fps: int,
    ) -> None:
        self._lane_names = lane_names
        self._lane_polygons = [np.array(p, dtype=np.int32) for p in lane_polygons]
        self._fps = fps
        self._lane_colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(lane_names))]
        self._lane_color_map = dict(zip(lane_names, self._lane_colors))

    def draw(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        lane_assignments: dict[str, list[TrackedObject]],
        metrics_snapshot: dict,
    ) -> np.ndarray:
        canvas = frame.copy()
        self._draw_lanes(canvas)
        self._draw_boxes(canvas, tracked_objects, lane_assignments)
        panel = self._draw_panel(frame.shape[0], metrics_snapshot)
        return np.hstack([canvas, panel])

    def _draw_lanes(self, canvas: np.ndarray) -> None:
        overlay = canvas.copy()
        for poly, color in zip(self._lane_polygons, self._lane_colors):
            cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        for poly, color in zip(self._lane_polygons, self._lane_colors):
            cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=2)

    def _draw_boxes(
        self,
        canvas: np.ndarray,
        tracked_objects: list[TrackedObject],
        lane_assignments: dict[str, list[TrackedObject]],
    ) -> None:
        # Build track_id -> lane color map
        id_to_color: dict[int, tuple[int, int, int]] = {}
        for lane_name, objects in lane_assignments.items():
            color = self._lane_color_map.get(lane_name, _GRAY)
            for obj in objects:
                id_to_color[obj.track_id] = color

        for obj in tracked_objects:
            color = id_to_color.get(obj.track_id, _GRAY)
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"#{obj.track_id} {obj.class_name}"
            cv2.putText(canvas, label, (x1, y1 - 8), _FONT, 0.5, color, 1, cv2.LINE_AA)

    def _draw_panel(self, height: int, snapshot: dict) -> np.ndarray:
        panel = np.zeros((height, _PANEL_WIDTH, 3), dtype=np.uint8)
        y = 30
        cv2.putText(panel, "Live Metrics", (10, y), _FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 35

        lanes_data = snapshot.get("lanes", {})
        for lane_name, color in zip(self._lane_names, self._lane_colors):
            data = lanes_data.get(lane_name, {})
            cv2.putText(panel, lane_name, (10, y), _FONT, 0.55, color, 1, cv2.LINE_AA)
            y += 22
            cv2.putText(panel, f"  Vehicles: {data.get('queue_length', 0)}", (10, y), _FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            y += 18
            total = data.get("throughput_total", 0)
            rate = data.get("throughput_rate", 0.0)
            cv2.putText(panel, f"  Throughput: {total} ({rate:.1f}/s)", (10, y), _FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            y += 18
            dwell = data.get("avg_dwell", 0.0)
            cv2.putText(panel, f"  Avg dwell: {dwell:.1f}s", (10, y), _FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            y += 18
            counts = data.get("vehicle_counts", {})
            if counts:
                parts = "  " + "  ".join(f"{k}: {v}" for k, v in counts.items())
                cv2.putText(panel, parts, (10, y), _FONT, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
            y += 25

        # Footer
        elapsed_frames = snapshot.get("elapsed_frames", 0)
        elapsed_sec = elapsed_frames / self._fps if self._fps > 0 else 0.0
        footer = f"FPS: {self._fps} | Elapsed: {elapsed_sec:.0f}s"
        cv2.putText(panel, footer, (10, height - 15), _FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        return panel
