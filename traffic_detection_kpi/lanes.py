from shapely.geometry import Point, Polygon

from traffic_detection_kpi import TrackedObject


class LaneZone:
    def __init__(self, name: str, polygon_coords: list[list[int]]):
        self.name = name
        self.polygon = Polygon(polygon_coords)

    def contains(self, point: tuple[int, int]) -> bool:
        return self.polygon.contains(Point(point))

    @classmethod
    def classify(
        cls, lanes: list["LaneZone"], tracked_objects: list[TrackedObject]
    ) -> dict[str, list[TrackedObject]]:
        result: dict[str, list[TrackedObject]] = {lane.name: [] for lane in lanes}
        for obj in tracked_objects:
            for lane in lanes:
                if lane.contains(obj.center):
                    result[lane.name].append(obj)
                    break  # first-match wins
        return result
