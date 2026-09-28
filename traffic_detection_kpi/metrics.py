from collections import defaultdict

from traffic_detection_kpi import TrackedObject, LaneMetrics, MetricsResult

# A track must occupy a lane for at least this long before its departure counts
# as a passage. Debounces bounding-box jitter across lane boundaries.
MIN_LANE_SECONDS = 0.2


class MetricsCollector:
    """Accumulates per-lane traffic metrics from frame-by-frame lane assignments.

    Throughput counts *completed passages*: a track is counted for a lane when it
    leaves that lane (moves to another lane, goes stale, or the run ends), provided
    it was present for at least ``min_lane_seconds``. A track is counted at most
    once per lane, so jitter across a boundary cannot inflate the total.
    """

    def __init__(
        self,
        lane_names: list[str],
        video_fps: int,
        max_age: int = 20,
        min_lane_seconds: float = MIN_LANE_SECONDS,
    ):
        self.lane_names = lane_names
        self.video_fps = video_fps
        self.max_age = max_age
        self.frame_count = 0
        self._min_lane_frames = (
            max(1, round(video_fps * min_lane_seconds)) if video_fps > 0 else 1
        )

        # Per-track state. _dwell_frames counts frames in the track's *current* lane.
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

    def _seconds(self, frames: int) -> float:
        return frames / self.video_fps if self.video_fps > 0 else 0.0

    def _close_passage(self, track_id: int, lane_name: str | None) -> None:
        """Record a completed passage through ``lane_name``, if it qualifies."""
        if lane_name is None or lane_name not in self._throughput:
            return
        if self._dwell_frames.get(track_id, 0) < self._min_lane_frames:
            return
        if track_id in self._counted_ids[lane_name]:
            return
        self._counted_ids[lane_name].add(track_id)
        self._throughput[lane_name] += 1
        self._vehicle_counts[lane_name][self._track_class.get(track_id, "unknown")] += 1

    def update(self, lane_assignments: dict[str, list[TrackedObject]]):
        self.frame_count += 1
        self._last_lane_assignments = lane_assignments

        lane_queue: dict[str, int] = {name: 0 for name in self.lane_names}
        lane_dwell_values: dict[str, list[float]] = {name: [] for name in self.lane_names}

        for lane_name, objects in lane_assignments.items():
            lane_queue[lane_name] = len(objects)
            for obj in objects:
                tid = obj.track_id
                previous_lane = self._track_lane.get(tid)
                if previous_lane is not None and previous_lane != lane_name:
                    # Lane change: close out the previous lane, restart dwell here.
                    self._close_passage(tid, previous_lane)
                    self._dwell_frames[tid] = 0

                self._dwell_frames[tid] = self._dwell_frames.get(tid, 0) + 1
                self._last_seen[tid] = self.frame_count
                self._track_lane[tid] = lane_name
                self._track_class[tid] = obj.class_name

                lane_dwell_values[lane_name].append(self._seconds(self._dwell_frames[tid]))

        # Prune stale tracks, closing out whatever lane they were last in.
        stale_ids = [
            tid for tid, last in self._last_seen.items()
            if self.frame_count - last > self.max_age
        ]
        for tid in stale_ids:
            self._close_passage(tid, self._track_lane.get(tid))
            self._dwell_frames.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._track_lane.pop(tid, None)
            self._track_class.pop(tid, None)

        # Sample time-series every second
        if self.video_fps > 0 and self.frame_count % self.video_fps == 0:
            for lane_name in self.lane_names:
                self._queue_ts[lane_name].append(lane_queue[lane_name])
                dwell_vals = lane_dwell_values[lane_name]
                avg_dwell = sum(dwell_vals) / len(dwell_vals) if dwell_vals else 0.0
                self._dwell_ts[lane_name].append(avg_dwell)

    def snapshot(self, elapsed_seconds: float | None = None) -> dict:
        """Return current per-lane metrics for live display.

        ``elapsed_seconds`` overrides the frame-derived duration; live sources pass
        wall-clock elapsed time, which is the only correct basis for a rate when
        processing cannot keep up with the stream.
        """
        duration = elapsed_seconds if elapsed_seconds is not None else self._seconds(self.frame_count)
        assignments = getattr(self, "_last_lane_assignments", {})

        lanes = {}
        for name in self.lane_names:
            objects = assignments.get(name, [])
            dwell_values = [
                self._seconds(self._dwell_frames.get(obj.track_id, 0)) for obj in objects
            ]

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

    def finalize(
        self,
        video_path: str = "",
        total_frames: int = 0,
        duration_seconds: float | None = None,
    ) -> MetricsResult:
        # Vehicles still in a lane when the run ends have their passage flushed.
        for tid, lane_name in list(self._track_lane.items()):
            self._close_passage(tid, lane_name)

        duration = duration_seconds if duration_seconds is not None else self._seconds(self.frame_count)
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
