#!/usr/bin/env python3


import argparse
import numpy as np
import gzip  #
import json
from datetime import datetime
from pathlib import Path


from amides.utils import read_json_file


def extract_amides_pt(entry: dict) -> float:
    try:
        return entry["amides"]
    except KeyError:
        return entry["processing_times"]["amides"]


def extract_pipeline_pt(entry: dict) -> float:
    try:
        return entry["pipeline"]
    except KeyError:
        return entry["processing_times"]["pipeline"]


def calculate_pt_metrics(pt_file_path: Path):
    amides_pts, pipeline_pts = [], []

    with gzip.open(str(pt_file_path), mode="r") as pt_file:
        for line in pt_file:
            entry = json.loads(line)
            amides_pts.append(extract_amides_pt(entry))
            pipeline_pts.append(extract_pipeline_pt(entry))

    avearage_pt_amides = np.average(amides_pts)
    std_pt_amides = np.std(amides_pts)
    average_pt_pipeline = np.average(pipeline_pts)
    std_pt_pipeline = np.std(pipeline_pts)

    print(f"Average PT AMIDES: {avearage_pt_amides:.8f} STD: {std_pt_amides:.8f}")
    print(f"Average PT Pipeline: {average_pt_pipeline:.8f} STD: {std_pt_pipeline:.8f}")


def calculate_event_processing_times(events_path: str):
    events_path = Path(events_path)
    if events_path.is_file():
        calculate_pt_metrics(events_path)
    else:
        event_files = events_path.glob("pt*.jsonl.gz")
        for event_file in event_files:
            calculate_pt_metrics(event_file)


def convert_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")


def calculate_performance_test_duration(wt_path: str):
    wall_times = read_json_file(wt_path)["wall_times"]
    durations = []

    for wall_time in wall_times:
        start = convert_timestamp(wall_time["start"])
        end = convert_timestamp(wall_time["end"])

        delta = end - start
        durations.append(delta.total_seconds())

    average_duration = np.average(durations)
    std_duration = np.std(durations)

    print(f"Average Duration: {average_duration:.8f} STD: {std_duration:.8f}")


def calculate_processing_time_metrics(args: argparse.Namespace):
    if args.events_file:
        calculate_event_processing_times(args.events_file)

    if args.wall_times:
        calculate_performance_test_duration(args.wall_times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events-file", action="store", help="Path to the performance metrics file(s)"
    )
    parser.add_argument(
        "--wall-times",
        action="store",
        help="JSON-file containing start and end timestamps of performance test runs",
    )
    args = parser.parse_args()
    calculate_processing_time_metrics(args)


if __name__ == "__main__":
    main()
