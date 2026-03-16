import logging

import cv2

from traffic_detection_kpi.config import Config
from traffic_detection_kpi.detection import YoloDetector
from traffic_detection_kpi.tracking import DeepSortTracker
from traffic_detection_kpi.lanes import LaneZone
from traffic_detection_kpi.metrics import MetricsCollector
from traffic_detection_kpi.reporting import ReportGenerator

logger = logging.getLogger(__name__)


class VideoPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.detector = YoloDetector(
            model_path=config.model.path,
            confidence=config.model.confidence,
            class_filter=config.model.classes,
        )
        self.tracker = DeepSortTracker(config.tracker)
        self.lanes = [LaneZone(l.name, l.polygon) for l in config.lanes]
        self.reporter = ReportGenerator(config.output_dir)
        self._fps: int | None = None

    def run(self):
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.config.video_path}")

        self._fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        metrics = MetricsCollector(
            lane_names=[l.name for l in self.lanes],
            video_fps=self._fps,
            max_age=self.config.tracker.max_age,
        )

        frame_num = 0
        consecutive_failures = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures < 5 and frame_num < total_frames - 1:
                    logger.warning(f"Corrupted frame at {frame_num}, seeking past it")
                    frame_num += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    continue
                break
            consecutive_failures = 0
            frame_num += 1

            detections = self.detector.detect(frame)
            tracked = self.tracker.track(detections, frame)
            assignments = LaneZone.classify(self.lanes, tracked)
            metrics.update(assignments)

            if frame_num % 100 == 0:
                logger.info(f"Processed {frame_num}/{total_frames} frames")

        cap.release()
        logger.info(f"Processing complete: {frame_num} frames")

        result = metrics.finalize(
            video_path=self.config.video_path,
            total_frames=frame_num,
        )
        self.reporter.generate(result)
        logger.info(f"Report saved to {self.config.output_dir}")
