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
    parser.add_argument("--youtube", metavar="URL", help="YouTube stream URL (grabs first frame)")
    parser.add_argument("--rtsp", metavar="URL", help="RTSP or RTMP stream URL (grabs first frame)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve video source for first frame
    if args.youtube:
        from traffic_detection_kpi.source import YouTubeSource
        source = YouTubeSource(args.youtube)
    elif args.rtsp:
        from traffic_detection_kpi.source import RtspSource
        source = RtspSource(args.rtsp)
    elif args.video or config.video_path:
        video_path = args.video or config.video_path
        source = None
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: cannot open video: {video_path}", file=sys.stderr)
            sys.exit(1)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error: cannot read first frame from video", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: no video source. Use --video, --youtube, --rtsp, or set video_path in config.", file=sys.stderr)
        sys.exit(1)

    if args.youtube or args.rtsp:
        ret, frame = source.read()
        source.release()
        if not ret:
            print("Error: cannot read first frame from stream", file=sys.stderr)
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
