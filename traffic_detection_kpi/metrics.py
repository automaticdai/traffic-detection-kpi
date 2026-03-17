from collections import defaultdict

from traffic_detection_kpi import TrackedObject, LaneMetrics, MetricsResult


class MetricsCollector:
    def __init__(self, lane_names: list[str], video_fps: int, max_age: int = 20):
        self.lane_names = lane_names
        self.video_fps = video_fps
        self.max_age = max_age
        self.frame_count = 0

        # Per-track state
        self._dwell_frames: dict[int, int] = {}
        self._last_seen: dict[int, int] = {}
        self._track_lane: dict[int, str] = {}
        self._track_class: dict[int, str] = {}

        # Per-lane accumulators
        self._throughput: dict[str, int] = {name: 0 for name in lane_names}
        self._counted_ids: dict[str, set[int]] = {name: set() for name in lane_names}
        self._vehicle_counts: dict[str, dict[str, int]] = {name: defaultdict(int) for name in lane_names}

        # Time-series
        self._queue_ts: dict[str, list[int]] = {name: [] for name in lane_names}
        self._dwell_ts: dict[str, list[float]] = {name: [] for name in lane_names}

    def update(self, lane_assignments: dict[str, list[TrackedObject]]):
        self.frame_count += 1
        self._last_lane_assignments = lane_assignments

        seen_this_frame: set[int] = set()
        lane_queue: dict[str, int] = {name: 0 for name in self.lane_names}
        lane_dwell_values: dict[str, list[float]] = {name: [] for name in self.lane_names}

        for lane_name, objects in lane_assignments.items():
            lane_queue[lane_name] = len(objects)
            for obj in objects:
                seen_this_frame.add(obj.track_id)
                self._last_seen[obj.track_id] = self.frame_count
                self._track_lane[obj.track_id] = lane_name
                self._track_class[obj.track_id] = obj.class_name

                if obj.track_id not in self._dwell_frames:
                    self._dwell_frames[obj.track_id] = 0
                self._dwell_frames[obj.track_id] += 1

                if (self._dwell_frames[obj.track_id] == self.video_fps
                        and obj.track_id not in self._counted_ids[lane_name]):
                    self._throughput[lane_name] += 1
                    self._counted_ids[lane_name].add(obj.track_id)
                    self._vehicle_counts[lane_name][obj.class_name] += 1

                dwell_seconds = self._dwell_frames[obj.track_id] / self.video_fps
                lane_dwell_values[lane_name].append(dwell_seconds)

        # Prune stale tracks
        stale_ids = [
            tid for tid, last in self._last_seen.items()
            if self.frame_count - last > self.max_age
        ]
        for tid in stale_ids:
            self._dwell_frames.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._track_lane.pop(tid, None)
            self._track_class.pop(tid, None)

        # Sample time-series every second
        if self.frame_count % self.video_fps == 0:
            for lane_name in self.lane_names:
                self._queue_ts[lane_name].append(lane_queue[lane_name])
                dwell_vals = lane_dwell_values[lane_name]
                avg_dwell = sum(dwell_vals) / len(dwell_vals) if dwell_vals else 0.0
                self._dwell_ts[lane_name].append(avg_dwell)

    def snapshot(self) -> dict:
        """Return current per-lane metrics for live display."""
        duration = self.frame_count / self.video_fps if self.video_fps > 0 else 0.0
        assignments = getattr(self, "_last_lane_assignments", {})

        lanes = {}
        for name in self.lane_names:
            objects = assignments.get(name, [])
            dwell_values = []
            for obj in objects:
                frames = self._dwell_frames.get(obj.track_id, 0)
                dwell_values.append(frames / self.video_fps if self.video_fps > 0 else 0.0)

            total = self._throughput[name]
            lanes[name] = {
                "queue_length": len(objects),
                "throughput_total": total,
                "throughput_rate": total / duration if duration > 0 else 0.0,
                "avg_dwell": sum(dwell_values) / len(dwell_values) if dwell_values else 0.0,
                "vehicle_counts": dict(self._vehicle_counts[name]),
            }

        return {
            "lanes": lanes,
            "elapsed_frames": self.frame_count,
        }

    def finalize(self, video_path: str = "", total_frames: int = 0) -> MetricsResult:
        duration = self.frame_count / self.video_fps if self.video_fps > 0 else 0.0
        lanes: dict[str, LaneMetrics] = {}
        for name in self.lane_names:
            total = self._throughput[name]
            lanes[name] = LaneMetrics(
                throughput_total=total,
                throughput_rate_avg=total / duration if duration > 0 else 0.0,
                vehicle_counts=dict(self._vehicle_counts[name]),
                queue_length_timeseries=list(self._queue_ts[name]),
                avg_dwell_time_timeseries=list(self._dwell_ts[name]),
            )
        return MetricsResult(
            video_path=video_path,
            total_frames=total_frames or self.frame_count,
            duration_seconds=duration,
            fps=self.video_fps,
            lanes=lanes,
        )
