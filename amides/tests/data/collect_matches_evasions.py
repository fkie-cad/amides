#! /usr/bin/env python3

import sys
import os
import logging
import json
import argparse

from amides.sigma import RuleSetDataset
from amides.events import benign_events_cache

base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../../"))

benign_events_dir = os.path.join(base_dir, "Daten/2021-02-05-socbed/split")
sigma_dir = os.path.join(base_dir, "Daten/Sigma-Studie")
pc_events_dir = os.path.join(sigma_dir, "events/windows/process_creation")
pc_rules_dir = os.path.join(sigma_dir, "rules/windows/process_creation")
out_dir = None

logger = logging.getLogger()


def load_events_and_pc_rules_data():
    try:
        pc_rules_dataset = RuleSetDataset(pc_events_dir, pc_rules_dir)
        return pc_rules_dataset
    except FileNotFoundError as err:
        logger.err(err)
        sys.exit(1)


def save_event(event, counter):
    file_name = f"{counter:06d}.json"
    
    with open(os.path.join(out_dir, file_name), "w") as out_file:
            json.dump(event, out_file, indent=4)


def save_matches_and_evasions(pc_rule_set_data):
    try:
        os.makedirs(out_dir, exist_ok=True)

        rule_datasets = pc_rule_set_data.rule_datasets.values()
        i = 1
        for rule_dataset in rule_datasets:
            for match in rule_dataset.matching_events.data:
                save_event(match, i)
                i += 1

            for evasion in rule_dataset.evasive_events.data:
                save_event(evasion, i)
                i += 1
                
    except OSError as err:
        logger.error(err)
        sys.exit(1)


def collect_matches_and_evasions():
    pc_rule_set_data = load_events_and_pc_rules_data()
    save_matches_and_evasions(pc_rule_set_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str, action="store", 
                        help="Path to the output directory")
    parser.add_argument("sigma_dir", type=str, nargs="?", action="store", 
                        help="Path to the directory where sigma_data is located")

    args = parser.parse_args()

    if args.sigma_dir:
        global sigma_dir
        sigma_dir = args.sigma_dir

    if args.out_dir:
        global out_dir
        out_dir = args.out_dir
    else:
        logger.error("No output directory specified. Exiting")
        sys.exit(1)

    collect_matches_and_evasions()



if __name__ == "__main__":
    main()
