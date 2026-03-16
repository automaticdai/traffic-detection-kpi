import json
from pathlib import Path

from traffic_detection_kpi import LaneMetrics, MetricsResult


def _sample_result() -> MetricsResult:
    return MetricsResult(
        video_path="test.mp4",
        total_frames=300,
        duration_seconds=10.0,
        fps=30,
        lanes={
            "Lane 1": LaneMetrics(
                throughput_total=5,
                throughput_rate_avg=0.5,
                vehicle_counts={"car": 3, "truck": 2},
                queue_length_timeseries=[1, 2, 3, 2, 1, 2, 3, 2, 1, 0],
                avg_dwell_time_timeseries=[0.5, 1.0, 1.5, 1.2, 0.8, 0.9, 1.1, 1.3, 0.7, 0.0],
            ),
            "Lane 2": LaneMetrics(
                throughput_total=3,
                throughput_rate_avg=0.3,
                vehicle_counts={"car": 2, "motorcycle": 1},
                queue_length_timeseries=[0, 1, 1, 2, 1, 0, 1, 1, 0, 0],
                avg_dwell_time_timeseries=[0.0, 0.5, 0.8, 1.0, 0.6, 0.0, 0.4, 0.7, 0.0, 0.0],
            ),
        },
    )


def test_generates_json(tmp_path):
    from traffic_detection_kpi.reporting import ReportGenerator
    gen = ReportGenerator(str(tmp_path))
    gen.generate(_sample_result())
    json_path = tmp_path / "metrics.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["fps"] == 30
    assert data["lanes"]["Lane 1"]["throughput_total"] == 5


def test_generates_chart_pngs(tmp_path):
    from traffic_detection_kpi.reporting import ReportGenerator
    gen = ReportGenerator(str(tmp_path))
    gen.generate(_sample_result())
    charts_dir = tmp_path / "charts"
    assert (charts_dir / "throughput_by_lane.png").exists()
    assert (charts_dir / "queue_length_over_time.png").exists()
    assert (charts_dir / "dwell_time_over_time.png").exists()
    assert (charts_dir / "vehicle_class_breakdown.png").exists()
