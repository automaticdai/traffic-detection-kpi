import logging
import os
import signal
import sys
import time

import cv2

from traffic_detection_kpi.config import Config
from traffic_detection_kpi.detection import YoloDetector
from traffic_detection_kpi.source import FileSource, VideoSource
from traffic_detection_kpi.tracking import DeepSortTracker
from traffic_detection_kpi.lanes import LaneZone
from traffic_detection_kpi.metrics import MetricsCollector
from traffic_detection_kpi.reporting import ReportGenerator
from traffic_detection_kpi.annotator import FrameAnnotator

logger = logging.getLogger(__name__)

WINDOW_NAME = "Traffic Detection KPI"


def _display_available() -> bool:
    """Best-effort check for a usable GUI.

    OpenCV's Qt backend calls abort() when it cannot reach a display, which Python
    cannot catch — so the environment must be checked *before* any window call.
    """
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


class VideoPipeline:
    def __init__(self, config: Config, source: VideoSource | None = None, show: bool = False):
        self.config = config
        self.source = source
        self.show = show
        self.detector = YoloDetector(
            model_path=config.model.path,
            confidence=config.model.confidence,
            class_filter=config.model.classes,
            class_ids=config.model.class_ids,
        )
        self.tracker = DeepSortTracker(config.tracker)
        self.lanes = [LaneZone(l.name, l.polygon) for l in config.lanes]
        self.reporter = ReportGenerator(config.output_dir)

    def _create_annotator(self, source: VideoSource) -> FrameAnnotator | None:
        """Open the overlay window, or return None if no GUI is usable."""
        if not _display_available():
            logger.warning(
                "No display detected (DISPLAY/WAYLAND_DISPLAY unset); disabling GUI overlay"
            )
            return None
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        except cv2.error:
            logger.warning("OpenCV has no GUI support available; disabling GUI overlay")
            return None
        return FrameAnnotator(
            lane_names=[l.name for l in self.lanes],
            lane_polygons=[l.polygon for l in self.config.lanes],
            fps=source.fps,
        )

    def run(self):
        source = self.source or FileSource(self.config.video_path)
        annotator = None
        shutdown = False
        prev_handler = None
        handler_installed = False

        if source.is_live:
            prev_handler = signal.getsignal(signal.SIGINT)

            def _handle_sigint(signum, frame):
                nonlocal shutdown
                logger.info("Shutdown requested, finishing current frame...")
                shutdown = True

            signal.signal(signal.SIGINT, _handle_sigint)
            handler_installed = True

        try:
            if self.show:
                annotator = self._create_annotator(source)
            self._run_loop(source, shutdown_check=lambda: shutdown, annotator=annotator)
        finally:
            if handler_installed:
                signal.signal(signal.SIGINT, prev_handler)
            source.release()
            if annotator:
                cv2.destroyAllWindows()

    def _run_loop(self, source: VideoSource, shutdown_check, annotator=None):
        metrics = MetricsCollector(
            lane_names=[l.name for l in self.lanes],
            video_fps=source.fps,
            max_age=self.config.tracker.max_age,
        )

        video_path = source.url
        frame_num = 0
        start_time = time.monotonic()

        while not shutdown_check():
            ret, frame = source.read()
            if not ret:
                break
            frame_num += 1

            detections = self.detector.detect(frame)
            tracked = self.tracker.track(detections, frame)
            assignments = LaneZone.classify(self.lanes, tracked)
            metrics.update(assignments)

            if annotator:
                try:
                    elapsed = time.monotonic() - start_time if source.is_live else None
                    snap = metrics.snapshot(elapsed_seconds=elapsed)
                    annotated = annotator.draw(frame, tracked, assignments, snap)
                    cv2.imshow(WINDOW_NAME, annotated)
                    delay = max(1, 1000 // source.fps)
                    if cv2.waitKey(delay) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    logger.warning("No display available, disabling GUI overlay")
                    annotator = None

            if frame_num % 100 == 0:
                if source.is_live:
                    logger.info(
                        "Processed %d frames (%.1fs elapsed)",
                        frame_num,
                        time.monotonic() - start_time,
                    )
                else:
                    logger.info("Processed %d frames", frame_num)

        logger.info("Processing complete: %d frames", frame_num)

        # A live stream's frame count says nothing about elapsed time when
        # processing cannot keep up, so rates must come off the wall clock.
        duration = time.monotonic() - start_time if source.is_live else None

        result = metrics.finalize(
            video_path=video_path,
            total_frames=frame_num,
            duration_seconds=duration,
        )
        self.reporter.generate(result)
        logger.info("Report saved to %s", self.config.output_dir)
