import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from traffic_detection_kpi import MetricsResult


class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / "charts"

    def generate(self, result: MetricsResult):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(result)
        self._chart_throughput(result)
        self._chart_queue_length(result)
        self._chart_dwell_time(result)
        self._chart_vehicle_classes(result)

    def _write_json(self, result: MetricsResult):
        data = asdict(result)
        path = self.output_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _chart_throughput(self, result: MetricsResult):
        sns.set_theme()
        names = list(result.lanes.keys())
        totals = [m.throughput_total for m in result.lanes.values()]
        fig, ax = plt.subplots()
        ax.bar(names, totals)
        ax.set_xlabel("Lane")
        ax.set_ylabel("Total vehicles")
        ax.set_title("Throughput by Lane")
        fig.savefig(self.charts_dir / "throughput_by_lane.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _chart_queue_length(self, result: MetricsResult):
        sns.set_theme()
        fig, ax = plt.subplots()
        for name, metrics in result.lanes.items():
            ax.plot(metrics.queue_length_timeseries, label=name)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Queue length")
        ax.set_title("Queue Length Over Time")
        ax.legend()
        fig.savefig(self.charts_dir / "queue_length_over_time.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _chart_dwell_time(self, result: MetricsResult):
        sns.set_theme()
        fig, ax = plt.subplots()
        for name, metrics in result.lanes.items():
            ax.plot(metrics.avg_dwell_time_timeseries, label=name)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Average dwell time (s)")
        ax.set_title("Dwell Time Over Time")
        ax.legend()
        fig.savefig(self.charts_dir / "dwell_time_over_time.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _chart_vehicle_classes(self, result: MetricsResult):
        sns.set_theme()
        totals: dict[str, int] = {}
        for metrics in result.lanes.values():
            for cls, count in metrics.vehicle_counts.items():
                totals[cls] = totals.get(cls, 0) + count
        if not totals:
            fig, ax = plt.subplots()
            ax.set_title("Vehicle Class Breakdown")
            fig.savefig(self.charts_dir / "vehicle_class_breakdown.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            return
        fig, ax = plt.subplots()
        ax.bar(list(totals.keys()), list(totals.values()))
        ax.set_xlabel("Vehicle class")
        ax.set_ylabel("Count")
        ax.set_title("Vehicle Class Breakdown")
        fig.savefig(self.charts_dir / "vehicle_class_breakdown.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
