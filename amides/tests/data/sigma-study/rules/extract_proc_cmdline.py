#! /usr/bin/env python3

import os
import re
import argparse
import json

from ruamel.yaml import YAML
from ruamel.yaml.composer import ComposerError

def read_yaml_file(file_path):
    try:
        yaml = YAML(typ="safe")
        with open(file_path, "r") as f:
            rule = yaml.load(f)
            return rule
    except ComposerError as exc:
        print(f"Error: {exc}")
        return None


def get_rule_filter(rule_dict):
    proc_cmdline_regex = r"process\.command_line:\s"

    rule_filter = rule_dict.get("filter", None)
    if rule_filter and re.search(proc_cmdline_regex, rule_filter):
        return rule_filter
    return None


def write_to_file(rule_filters, out_file):
    with open(out_file, "w") as f:
        json.dump(rule_filters, f, indent=1, sort_keys=True)


def show_results(rule_filters):
    print(json.dumps(rule_filters, indent=1, sort_keys=True))


def extract_process_cmdline(rules_path, out_file=None):
    rule_filters = {}

    for rule_name in os.listdir(rules_path):
        rule_path = os.path.join(rules_path, rule_name)
        if os.path.isfile(rule_path) and (rule_name.endswith(".yaml") or rule_name.endswith(".yml")):
            rule_dict = read_yaml_file(rule_path)
            if rule_dict:
                filter = get_rule_filter(rule_dict)
                if filter:
                    rule_filters[rule_name] = filter

    if out_file:
        write_to_file(rule_filters, out_file)
    else:
        show_results(rule_filters)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-r", "--rules-path", action="store",
                           help="Path to process_creation rules directory")
    argparser.add_argument("-o", "--out-file", action="store",
                           help="Output file for results")
    args = argparser.parse_args()

    if args.rules_path:
        extract_process_cmdline(args.rules_path, args.out_file)
