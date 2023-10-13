#! /usr/bin/env python3

import sys
import os
import json
import argparse
import logging
import json

base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../"))

events_dir = os.path.join(
    base_dir, "../Daten/2021-02-05-socbed/split/Microsoft-Windows-Sysmon_1"
)
out_file = None
logger = logging.getLogger()


def load_events():
    try:
        events = []
        entries = os.listdir(events_dir)
        for entry in entries:
            with open(os.path.join(events_dir, entry), "r") as event_file:
                events.append(json.load(event_file))

        return events
    except FileNotFoundError as err:
        logger.error(err)
        sys.exit(1)


def save_events(events):
    try:
        with open(out_file, mode="w") as out:
            for event in events:
                out.write(f"{json.dumps(event)}\n")
    except OSError as err:
        logger.error(err)
        sys.exit(1)


def convert_json_to_jsonl():
    events = load_events()
    save_events(events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out_file", type=str, action="store", help="Path to the output directory"
    )
    parser.add_argument(
        "events_dir",
        type=str,
        nargs="?",
        action="store",
        help="Path to the directory where event data is located",
    )

    args = parser.parse_args()

    if args.events_dir:
        global events_dir
        events_dir = args.events_dir

    if args.out_file:
        global out_file
        out_file = args.out_file
    else:
        logger.error("No output directory specified. Exiting")
        sys.exit(1)

    convert_json_to_jsonl()


if __name__ == "__main__":
    main()
