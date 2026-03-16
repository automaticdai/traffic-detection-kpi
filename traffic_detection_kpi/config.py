from dataclasses import dataclass, field
from pathlib import Path

import yaml
from shapely.geometry import Polygon as ShapelyPolygon


COCO_CLASS_MAP = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


@dataclass
class LaneConfig:
    name: str
    polygon: list[list[int]]


@dataclass
class ModelConfig:
    path: str
    confidence: float
    classes: list[str]
    class_ids: list[int] = field(default_factory=list)


@dataclass
class TrackerConfig:
    type: str
    max_age: int
    n_init: int
    max_cosine_distance: float
    embedder: str


@dataclass
class Config:
    video_path: str | None
    output_dir: str
    model: ModelConfig
    tracker: TrackerConfig
    lanes: list[LaneConfig]


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    _validate_required(raw, ["output_dir", "model", "tracker", "lanes"])

    model_raw = raw["model"]
    _validate_required(model_raw, ["path", "confidence", "classes"])
    for cls in model_raw["classes"]:
        if cls not in COCO_CLASS_MAP:
            raise ValueError(f"Unknown class name: '{cls}'. Valid: {list(COCO_CLASS_MAP.keys())}")
    class_ids = [COCO_CLASS_MAP[c] for c in model_raw["classes"]]
    model = ModelConfig(
        path=model_raw["path"],
        confidence=model_raw["confidence"],
        classes=model_raw["classes"],
        class_ids=class_ids,
    )

    tracker_raw = raw["tracker"]
    _validate_required(tracker_raw, ["type", "max_age", "n_init", "max_cosine_distance", "embedder"])
    tracker = TrackerConfig(**{k: tracker_raw[k] for k in ["type", "max_age", "n_init", "max_cosine_distance", "embedder"]})

    lanes = []
    for i, lane_raw in enumerate(raw["lanes"]):
        _validate_required(lane_raw, ["name", "polygon"])
        polygon = lane_raw["polygon"]
        if len(polygon) < 3:
            raise ValueError(f"Lane '{lane_raw['name']}' polygon must have at least 3 points, got {len(polygon)}")
        if not ShapelyPolygon(polygon).is_valid:
            raise ValueError(f"Lane '{lane_raw['name']}' polygon is self-intersecting")
        lanes.append(LaneConfig(name=lane_raw["name"], polygon=polygon))

    return Config(
        video_path=raw.get("video_path"),
        output_dir=raw["output_dir"],
        model=model,
        tracker=tracker,
        lanes=lanes,
    )


def _validate_required(data: dict, keys: list[str]):
    for key in keys:
        if key not in data:
            raise ValueError(f"Missing required field: '{key}'")
