#!/usr/bin/env python3

import json
import os
import sys
import argparse

from amides.persist import Dumper
from amides.sigma import RuleSetDataset
from amides.models.baseline.baseline import BaselineClassifier
from amides.utils import get_logger, set_log_level, read_json_file

set_log_level("info")
logger = get_logger(__name__)


base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../"))
sigma_dir = os.path.join(base_dir, "Daten/Sigma-Studie")
pc_events_dir = os.path.join(sigma_dir, "events/windows/process_creation")
pc_rules_dir = os.path.join(sigma_dir, "rules/windows/process_creation")
benign_events_path = os.path.join(base_dir, "Daten/2021-02-05-socbed/benign.jsonl")

dumper = None
output_dir = None

save = False

remove_escape_characters = False
delete_whitespaces = False
remove_exe = False
add_exe = False
swap_minus_slash = False
swap_slash_minus = False

iterative = False


def init_dumper():
    global dumper

    try:
        if not dumper:
            dumper = Dumper(output_dir)
    except OSError as err:
        logger.error(err)
        sys.exit(1)


def load_pc_rules_data():
    try:
        return RuleSetDataset(pc_events_dir, pc_rules_dir)
    except FileNotFoundError as err:
        logger.error(err)
        sys.exit(1)


def benign_events():
    for event in open(benign_events_path, "r"):
        yield json.loads(event)


def evaluate_baseline_classifier():
    pc_rules_data = load_pc_rules_data()

    modifier_config = {
        "remove_escape_characters": remove_escape_characters,
        "delete_whitespaces": delete_whitespaces,
        "remove_exe": remove_exe,
        "add_exe": add_exe,
        "swap_minus_slash": swap_minus_slash,
        "swap_slash_minus": swap_slash_minus,
    }

    baseline_clf = BaselineClassifier(
        modifier_config, pc_rules_dir, iterative=iterative
    )
    baseline_clf.evaluate_benign_events(benign_events())
    baseline_clf.reset_modifier_mask()

    evasive_events = pc_rules_data.evasive_events
    baseline_clf.evaluate_malicious_events(evasive_events.data)

    if save:
        dumper.save_object(baseline_clf.results, baseline_clf.file_name())


def parse_args_and_options(args):
    if args.benign_events_path:
        global benign_events_path
        benign_events_path = args.benign_events_path

    if args.sigma_dir:
        global sigma_dir
        global pc_events_dir
        global pc_rules_dir

        sigma_dir = args.sigma_dir
        pc_events_dir = os.path.join(sigma_dir, "events/windows/process_creation")
        pc_rules_dir = os.path.join(sigma_dir, "rules/windows/process_creation")

    if args.save:
        global save
        global output_dir
        save = args.save

        if args.out_dir:
            output_dir = args.out_dir
        else:
            output_dir = os.path.join(os.getcwd(), "models")
            logger.warning(
                "No output dir for results data specified. Using %s", output_dir
            )

        init_dumper()

    if args.iterative:
        global iterative
        iterative = True

    if args.remove_escape_chars:
        global remove_escape_characters
        remove_escape_characters = True

    if args.delete_whitespaces:
        global delete_whitespaces
        delete_whitespaces = True

    if args.add_exe:
        global add_exe
        add_exe = True

    if args.remove_exe:
        global remove_exe
        remove_exe = True

    if args.swap_minus_slash:
        global swap_minus_slash
        swap_minus_slash = True

    if args.swap_slash_minus:
        global swap_slash_minus
        swap_slash_minus = True


def load_args_from_file(parser, config_path):
    config_dict = read_json_file(config_path)
    if config_dict is None:
        logger.error("Error reading config file. Exiting")
        sys.exit(1)

    ns = argparse.Namespace()
    ns.__dict__.update(config_dict)
    args = parser.parse_args(namespace=ns)

    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benign-events-path", type=str, action="store", help="Path to benign events"
    )
    parser.add_argument(
        "--sigma-dir",
        type=str,
        action="store",
        help="Path to the directory where Sigma data is located",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        action="store",
        help="Specifies the path of the output directory where results should be saved",
    )
    parser.add_argument(
        "--save", action="store_true", help="Specifies if output should be saved"
    )
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="If baseline classifier should be evaluated in iterative mode",
    )
    parser.add_argument(
        "--remove-escape-chars",
        action="store_true",
        help="Activate modifier: Remove escape characters",
    )
    parser.add_argument(
        "--delete-whitespaces",
        action="store_true",
        help="Activate modifier: Deletes whitespaces",
    )
    parser.add_argument(
        "--remove-exe",
        action="store_true",
        help="Activate modifier: Removes '.exe' ending from binary names",
    )
    parser.add_argument(
        "--add-exe",
        action="store_true",
        help="Activate modifier: Adds '.exe' ending to binary names",
    )
    parser.add_argument(
        "--swap-minus-slash",
        action="store_true",
        help="Activate modifier: Swaps all '-' symbols with '/' symbols",
    )
    parser.add_argument(
        "--swap-slash-minus",
        action="store_true",
        help="Activate modifier: Swaps all '/' with '-' symbols",
    )
    parser.add_argument(
        "--config", type=str, action="store", help="Path to config file."
    )

    args = parser.parse_args()

    if args.config:
        args = load_args_from_file(parser, args.config)

    parse_args_and_options(args)
    evaluate_baseline_classifier()


if __name__ == "__main__":
    main()
