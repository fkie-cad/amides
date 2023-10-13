from luqum.parser import parser
from amides.features.normalize import Normalizer
from luqum.visitor import TreeVisitor
from luqum.tree import NoneItem
from argparse import ArgumentParser
import glob
import os
import yaml
import itertools


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
    rule_filters = [doc["filter"] for doc in read_rules(args.dir)]
    terms_per_filter = [terms_of_interest(filter, fields_of_interest) for filter in rule_filters]
    print_terms(terms_per_filter, args.normalize)


def parse_args():
    parser = ArgumentParser(description="Extract search terms from converted Sigma rules.")
    parser.add_argument("--normalize", action="store_true", help="normalize output")
    parser.add_argument("dir", help="rule directory")
    parser.add_argument("fields", help="comma-separated list of field names")
    return parser.parse_args()


def read_rules(dir):
    docs = []
    rule_files = glob.glob(os.path.join(dir, "*.yml"))
    for rule_file in rule_files:
        with open(rule_file) as f:
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
