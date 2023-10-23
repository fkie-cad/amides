#!/usr/bin/env python3
"""This script enables to  extract values of different search fields from Sigma rules. Extracted search field values can be additionally
normalized, i.e. split into lists of token strings using preprocessing, tokenization, and token elimination.
"""


import glob
import os
import yaml
import itertools
from luqum.parser import parser
from argparse import ArgumentParser

from amides.features.normalize import Normalizer
from amides.sigma import MultiFieldVisitor


def main():
    args = parse_args()
    fields_of_interest = args.fields.split(",")
    rule_filters = [doc["filter"] for doc in read_rules(args.dir)]
    terms_per_filter = [
        terms_of_interest(filter, fields_of_interest) for filter in rule_filters
    ]
    print_terms(terms_per_filter, args.normalize)


def parse_args():
    parser = ArgumentParser(
        description="Extract search terms from converted Sigma rules."
    )
    parser.add_argument("--normalize", action="store_true", help="normalize output")
    parser.add_argument("dir", help="rule directory")
    parser.add_argument("fields", help="comma-separated list of field names")
    return parser.parse_args()


def read_rules(dir):
    docs = []
    rule_files = glob.glob(os.path.join(dir, "*.yml"))
    for rule_file in rule_files:
        with open(rule_file, "r", encoding="utf-8") as f:
            docs.extend(yaml.safe_load_all(f))
    return docs


def terms_of_interest(filter, fields):
    tree = parser.parse(filter)
    visitor = MultiFieldVisitor(fields)
    visitor.visit(tree)
    return visitor.values


def print_terms(terms_per_filter, normalize):
    normalizer = Normalizer(max_len_num_values=3)
    terms_flat = itertools.chain(*terms_per_filter)
    for term in terms_flat:
        output = normalizer.normalize(term) if normalize else term
        if output:  # Avoids empty vectors in training set
            print(output)


if __name__ == "__main__":
    main()
