from dataclasses import dataclass, field


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x1, y1, w, h (xywh)
    class_id: int
    class_name: str
    confidence: float


@dataclass
class TrackedObject:
    track_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (ltrb)
    class_id: int
    class_name: str
    center: tuple[int, int]  # midpoint of ltrb bbox


@dataclass
class LaneMetrics:
    throughput_total: int = 0
    throughput_rate_avg: float = 0.0
    vehicle_counts: dict[str, int] = field(default_factory=dict)
    queue_length_timeseries: list[int] = field(default_factory=list)
    avg_dwell_time_timeseries: list[float] = field(default_factory=list)


@dataclass
class MetricsResult:
    video_path: str
    total_frames: int
    duration_seconds: float
    fps: int
    lanes: dict[str, LaneMetrics] = field(default_factory=dict)
