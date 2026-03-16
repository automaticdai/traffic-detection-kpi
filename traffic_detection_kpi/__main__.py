import argparse
import logging
import sys

from traffic_detection_kpi.config import load_config
from traffic_detection_kpi.pipeline import VideoPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Detection KPI — offline video analytics"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
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
    pipeline = VideoPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
