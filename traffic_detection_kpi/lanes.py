from shapely.geometry import Point, Polygon, box

from traffic_detection_kpi import TrackedObject

MIN_OVERLAP_RATIO = 0.25


class LaneZone:
    def __init__(self, name: str, polygon_coords: list[list[int]]):
        self.name = name
        self.polygon = Polygon(polygon_coords)

    def contains(self, point: tuple[int, int]) -> bool:
        return self.polygon.contains(Point(point))

    def overlap_ratio(self, bbox: tuple[int, int, int, int]) -> float:
        """Return fraction of bbox area that overlaps with this lane polygon."""
        x1, y1, x2, y2 = bbox
        bbox_poly = box(x1, y1, x2, y2)
        bbox_area = bbox_poly.area
        if bbox_area <= 0:
            return 0.0
        intersection = self.polygon.intersection(bbox_poly).area
        return intersection / bbox_area

    @classmethod
    def classify(
        cls,
        lanes: list["LaneZone"],
        tracked_objects: list[TrackedObject],
        min_overlap: float = MIN_OVERLAP_RATIO,
    ) -> dict[str, list[TrackedObject]]:
        result: dict[str, list[TrackedObject]] = {lane.name: [] for lane in lanes}
        for obj in tracked_objects:
            best_lane = None
            best_overlap = 0.0
            for lane in lanes:
                ratio = lane.overlap_ratio(obj.bbox)
                if ratio >= min_overlap and ratio > best_overlap:
                    best_overlap = ratio
                    best_lane = lane
            if best_lane is not None:
                result[best_lane.name].append(obj)
        return result
