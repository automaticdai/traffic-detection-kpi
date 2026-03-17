import argparse
import sys

import cv2
import yaml

from traffic_detection_kpi.annotator import LANE_PALETTE
from traffic_detection_kpi.config import load_config
from traffic_detection_kpi.lane_editor import LaneEditor


def save_lanes_to_config(config_path: str, lanes: list[dict]) -> None:
    """Overwrite the lanes section of a YAML config file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw["lanes"] = lanes
    with open(config_path, "w") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive lane polygon editor"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--video", help="Path to video file (overrides config video_path)")
    args = parser.parse_args()

    config = load_config(args.config)
    video_path = args.video or config.video_path
    if not video_path:
        print("Error: no video source. Use --video or set video_path in config.", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: cannot read first frame from video", file=sys.stderr)
        sys.exit(1)

    lanes = [{"name": l.name, "polygon": l.polygon} for l in config.lanes]
    colors = [LANE_PALETTE[i % len(LANE_PALETTE)] for i in range(len(lanes))]

    modified, updated_lanes = LaneEditor(frame, lanes, colors).run()

    if modified:
        save_lanes_to_config(args.config, updated_lanes)
        print(f"Saved {len(updated_lanes)} lanes to {args.config}")
    else:
        print("No changes saved.")


if __name__ == "__main__":
    main()
