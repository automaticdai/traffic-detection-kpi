from deep_sort_realtime.deepsort_tracker import DeepSort

from traffic_detection_kpi import Detection, TrackedObject
from traffic_detection_kpi.config import TrackerConfig, COCO_CLASS_MAP


class DeepSortTracker:
    def __init__(self, config: TrackerConfig):
        self.tracker = DeepSort(
            max_age=config.max_age,
            n_init=config.n_init,
            nms_max_overlap=0.3,
            max_cosine_distance=config.max_cosine_distance,
            nn_budget=None,
            override_track_class=None,
            embedder=config.embedder,
            half=True,
            bgr=True,
        )

    def track(self, detections: list[Detection], frame) -> list[TrackedObject]:
        # Convert Detection to DeepSort format: ([x, y, w, h], confidence, class_name)
        ds_detections = []
        for det in detections:
            ds_detections.append((list(det.bbox), det.confidence, det.class_name))

        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        result = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Get class info from the track's detection
            det = track.get_det_class()
            class_name = str(det) if det is not None else "unknown"
            class_id = COCO_CLASS_MAP.get(class_name, -1)

            result.append(TrackedObject(
                track_id=track.track_id,
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=class_name,
                center=center,
            ))
        return result
