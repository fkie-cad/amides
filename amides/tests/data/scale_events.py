#! /usr/bin/env python3

import sys
import os
import json
import argparse
import random
import logging

base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../"))

events_dir = os.path.join(base_dir, "../Daten/2021-02-05-socbed/split/Microsoft-Windows-Sysmon_1")
out_dir = None
scale = 10.0

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
        os.makedirs(out_dir, exist_ok=True)
        for i in range(len(events)):
            file_name = f"{i:06d}.json"
            with open(os.path.join(out_dir, file_name), "w") as out_file:
                json.dump(events[i], out_file, indent=4)
    except OSError as err:
        logger.error(err)
        sys.exit(1)


def scale_events():
    events = load_events()
    if not events:
        logger.error("No events in specified directory. Exiting")
        sys.exit(1)

    requested_num_events = int(len(events) * scale)
    scaled_events = random.choices(events, k=requested_num_events)

    save_events(scaled_events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str, action="store", 
                        help="Path to the output directory")
    parser.add_argument("events_dir", type=str, nargs="?", action="store", 
                        help="Path to the directory where event data is located")
    parser.add_argument("--scale", type=int, action="store",
                        help="Scaling factor (Default: 10)")

    args = parser.parse_args()

    if args.events_dir:
        global events_dir
        events_dir = args.events_dir

    if args.out_dir:
        global out_dir
        out_dir = args.out_dir
    else:
        logger.error("No output directory specified. Exiting")
        sys.exit(1)

    if args.scale:
        global scale
        scale = args.scale

    scale_events()


if __name__ == "__main__":
    main()