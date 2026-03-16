import argparse
import logging
import sys

from traffic_detection_kpi.config import load_config
from traffic_detection_kpi.pipeline import VideoPipeline
from traffic_detection_kpi.source import FileSource, YouTubeSource, RtspSource, VideoSource


def _build_source(args, config) -> VideoSource:
    cli_sources = []
    if args.youtube:
        cli_sources.append(("youtube", args.youtube))
    if args.rtsp:
        cli_sources.append(("rtsp", args.rtsp))

    if len(cli_sources) > 1:
        print("Error: only one of --youtube or --rtsp may be specified", file=sys.stderr)
        sys.exit(1)

    if cli_sources:
        kind, url = cli_sources[0]
        if kind == "youtube":
            return YouTubeSource(url)
        else:
            return RtspSource(url)

    if config.video_path:
        return FileSource(config.video_path)

    print("Error: no video source provided. Use --youtube, --rtsp, or set video_path in config.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Detection KPI — video analytics"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--youtube", metavar="URL", help="YouTube live stream URL"
    )
    parser.add_argument(
        "--rtsp", metavar="URL", help="RTSP or RTMP stream URL"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    source = _build_source(args, config)
    pipeline = VideoPipeline(config, source=source)
    pipeline.run()


if __name__ == "__main__":
    main()
