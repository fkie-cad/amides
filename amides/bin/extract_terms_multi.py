#!/usr/bin/env python3
"""This script enables to  extract values of different search fields from Sigma rules. Extracted search field values can be additionally
normalized, i.e. split into lists of token strings using preprocessing, tokenization, and token elimination.
"""

import glob
import os
import yaml
import json

from luqum.parser import parser
from amides.features.normalize import Normalizer
from amides.sigma import MultiFieldVisitor
from argparse import ArgumentParser
from pathlib import Path


def main():
    args = parse_args()
    fields_of_interest = args.fields.split(",")

    rule_files = glob.glob(os.path.join(args.dir, "*.yml"))
    normalizer = Normalizer(max_len_num_values=3)
    rules = []
    for rule_file in rule_files:
        with open(rule_file, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
            rule = {}
            # rule["title"] = docs[0]["pre_detector"]["title"]
            rule["title"] = Path(rule_file).stem
            samples = []
            for doc in docs:
                terms = terms_of_interest(doc["filter"], fields_of_interest)
                samples.extend(terms)
            normalized_samples = []
            for sample in samples:
                normalized_sample = normalizer.normalize(sample)
                if normalized_sample:
                    normalized_samples.append(normalized_sample)
            rule["samples"] = normalized_samples
            rules.append(rule)

    for rule in rules:
        print(json.dumps(rule))


def parse_args():
    parser = ArgumentParser(
        description="Extract search terms from converted Sigma rules."
    )
    parser.add_argument("dir", help="rule directory")
    parser.add_argument("fields", help="comma-separated list of field names")
    return parser.parse_args()


def terms_of_interest(filter, fields):
    tree = parser.parse(filter)
    visitor = MultiFieldVisitor(fields)
    visitor.visit(tree)
    return visitor.values


if __name__ == "__main__":
    main()
