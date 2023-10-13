#! /usr/bin/env python3

import argparse
import sys
import os
import json

from amides.sigma import RuleSetDataset
from amides.utils import set_log_level, get_logger
from amides.features.tokenization import AnyWordCharacter
from amides.features.preprocessing import FilterDummyCharacters, Lowercase
from amides.features.filter import NumericValues, Strings
from amides.features.extraction import CommandlineExtractor

set_log_level("info")
_logger = get_logger("tokenize")

base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../"))

# Preprocessor
filter_dummy = FilterDummyCharacters()
lower = Lowercase()

# Tokenizer
any_word = AnyWordCharacter()

# TokenEliminator
hex_values = NumericValues(length=3)
strings = Strings(length=30)


def preprocessor(string):
    filtered = filter_dummy(string)

    return lower(filtered)


def tokenizer(string):
    tokens = any_word(string)

    return tokens


def token_elimination(token_list):
    return strings(hex_values(token_list))


def analyzer(string):
    preprocessed = preprocessor(string)
    token_list = tokenizer(preprocessed)
    return token_elimination(token_list)


def load_rules_data(sigma_dir):
    events_dir = os.path.join(sigma_dir, "events/windows/process_creation")
    rules_dir = os.path.join(sigma_dir, "rules/windows/process_creation")

    try:
        return RuleSetDataset(events_dir, rules_dir)
    except FileNotFoundError as err:
        _logger.error(err)
        sys.exit(1)


def extract_tokens(cmdline):
    tokens = []

    for token in analyzer(cmdline):
        tokens.append(token)

    return tokens


def samples_file(samples_file_path: str):
    try:
        with open(samples_file_path, "r") as in_file:
            for line in in_file:
                stripped = line.rstrip("\n")

                yield json.loads(stripped)
    except FileNotFoundError as err:
        _logger.error(err)
        sys.exit(1)


def event_cmdlines(events):
    for event in events.data:
        yield CommandlineExtractor.extract_commandline_from_event(event)


def create_normalized_samples_file(
    line_generator: callable, num_samples: int, out_file: str
):
    with open(out_file, "w", encoding="utf-8") as out:
        for i, line in enumerate(line_generator):
            if i == num_samples:
                break

            tokens = extract_tokens(line)
            if len(tokens) == 0:
                continue

            tokens.sort()
            tokens_csv = ",".join(tokens)
            out.write(f"{tokens_csv}\n")


def normalize_samples(samples_file_path: str, num_samples: int, out_file: str):
    create_normalized_samples_file(
        samples_file(samples_file_path), num_samples, out_file
    )


def normalize_rule_filters(sigma_dir: str, num_samples: int, out_file: str):
    rule_set_data = load_rules_data(sigma_dir)

    create_normalized_samples_file(
        rule_set_data.extract_field_values_from_filter(
            search_fields=["proceess.command_line"]
        ),
        num_samples,
        out_file,
    )


def normalize_evasions(sigma_dir: str, num_samples: int, out_file: str):
    rule_set_data = load_rules_data(sigma_dir)

    create_normalized_samples_file(
        event_cmdlines(rule_set_data.evasions), num_samples, out_file
    )


def normalize_matches(sigma_dir: str, num_samples: int, out_file: str):
    rule_set_data = load_rules_data(sigma_dir)

    create_normalized_samples_file(
        event_cmdlines(rule_set_data.matches), num_samples, out_file
    )


def normalize(args):
    if args.samples_file is not None:
        normalize_samples(args.samples_file, args.num_samples, args.out_file)
    elif args.rule_filters:
        normalize_rule_filters(args.sigma_dir, args.num_samples, args.out_file)
    elif args.evasions:
        normalize_evasions(args.sigma_dir, args.num_samples, args.out_file)
    elif args.matches:
        normalize_matches(args.sigma_dir, args.num_samples, args.out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples_file", type=str, nargs="?", action="store")
    parser.add_argument("--rule-filters", action="store_true", default=False)
    parser.add_argument("--evasions", action="store_true", default=False)
    parser.add_argument("--matches", action="store_true", default=False)
    parser.add_argument("--num-samples", action="store_true", default=-1)
    parser.add_argument(
        "--sigma-dir",
        action="store",
        default=os.path.join(base_dir, "Daten/Sigma-Studie"),
    )
    parser.add_argument(
        "-o",
        "--out-file",
        type=str,
        action="store",
        default=os.path.join(os.getcwd(), "tokens.out"),
    )

    args = parser.parse_args()

    normalize(args)


if __name__ == "__main__":
    main()
