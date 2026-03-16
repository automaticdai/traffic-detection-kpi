from ultralytics import YOLO

from traffic_detection_kpi import Detection


class YoloDetector:
    def __init__(self, model_path: str, confidence: float, class_filter: list[str]):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.allowed_names = set(class_filter)

    def detect(self, frame) -> list[Detection]:
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        result = results[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = result.names.get(class_id, "")
            if class_name not in self.allowed_names:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0].item())
            detections.append(Detection(
                bbox=(x1, y1, w, h),
                class_id=class_id,
                class_name=class_name,
                confidence=conf,
            ))
        return detections
