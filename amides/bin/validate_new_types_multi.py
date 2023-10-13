import argparse
import re
from pprint import pprint

from amides.features.normalize import Normalizer
from amides.persist import Dumper


def main():
    args = parse_args()
    filename_regex = ".*multi_train_rslt_([a-z]+)_[0-9]+_[0-9]+.zip"
    category = re.compile(filename_regex).match(args.file).group(1)

    evasion_values = open("data/generalisation/values_evasion_" + category).readlines()
    normalizer = Normalizer(max_len_num_values=3)
    evasion_tokens = [normalizer.normalize(sample) for sample in evasion_values]

    dumper = Dumper()
    multi_train_result = dumper.load_object(args.file)

    attributions = [{}, {}, {}]  # We only have three evasions

    for result in multi_train_result.results.values():
        feature_extractor = result.feature_extractors[0]
        vectors = feature_extractor.transform(test_samples(evasion_tokens))
        predict = result.estimator.decision_function(vectors)

        for i in range(3):
            attributions[i][result.name] = predict[i]

    for i in range(3):
        print(evasion_values[i])
        pprint(sorted(attributions[i].items(), key=lambda item: item[1], reverse=True))
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run rule attribution with trained classifiers."
    )
    parser.add_argument("file", help="zip file containing training results")
    return parser.parse_args()


def test_samples(evasion_tokens):
    for sample in evasion_tokens:
        yield sample


if __name__ == "__main__":
    main()
