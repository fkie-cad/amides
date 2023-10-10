#!/usr/bin/env python3

from luqum.parser import parser
from amides.features.normalize import Normalizer
from luqum.visitor import TreeVisitor
from luqum.tree import NoneItem
from argparse import ArgumentParser
from pathlib import Path

import glob
import os
import yaml
import json


class MultiFieldVisitor(TreeVisitor):
    def __init__(self, fields):
        super(MultiFieldVisitor, self).__init__(track_parents=False)
        self._fields = fields
        self._values = []

    @property
    def values(self):
        return self._values

    def visit_search_field(self, node, context):
        match = False
        for field in self._fields:
            if node.name == field or node.name.startswith(field + "|"):
                match = True
        if match:
            context = self.child_context(node, NoneItem(), context)
            context[node.name.split("|")[0]] = True
        yield from self.generic_visit(node, context)

    def visit_phrase(self, node, context):
        for field in self._fields:
            if context.get(field, False):
                if node.value.startswith('"') and node.value.endswith('"'):
                    self._values.append(node.value[1:-1])
                else:
                    self._values.append(node.value)
        yield from self.generic_visit(node, context)

    def visit_not(self, node, context):
        yield NoneItem()


def main():
    args = parse_args()
    fields_of_interest = args.fields.split(",")

    rule_files = glob.glob(os.path.join(args.dir, "*.yml"))
    normalizer = Normalizer(max_len_num_values=3)
    rules = []
    for rule_file in rule_files:
        with open(rule_file) as f:
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
