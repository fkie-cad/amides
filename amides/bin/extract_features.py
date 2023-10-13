#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np

from amides.utils import get_logger, set_log_level
from amides.persist import Dumper
from amides.visualization import WeightsFeaturesPlot


set_log_level("info")
_logger = get_logger("extract-features")

dumper = None
num_top_weights = 20


def init_dumper(out_dir):
    global dumper

    try:
        if not dumper:
            dumper = Dumper(out_dir)
    except OSError as err:
        _logger.error(err)
        sys.exit(1)


def write_to_file(result, features, out_dir):
    feature_file_name = f"weights_features_{result.name}_{result.timestamp}.txt"
    feature_file_path = f"{out_dir}/{feature_file_name}"

    with open(feature_file_path, "w", encoding="utf-8") as vocab_file:
        for feature in features:
            vocab_file.write(f"{tuple(feature)}\n")


def get_non_zero_features(weights, features):
    non_zero_idcs = np.where((weights > 0.0) | (weights < 0.0))[0]
    non_zero_weights = weights[non_zero_idcs]
    non_zero_features = features[non_zero_idcs]

    return non_zero_weights, non_zero_features


def get_top_features(weights, features):
    top_positive_weights = weights[-num_top_weights:]
    top_negative_weights = weights[:num_top_weights]

    top_positive_features = features[-num_top_weights:]
    top_negative_features = features[:num_top_weights]

    top_weights = np.hstack((top_negative_weights, top_positive_weights))
    top_features = np.hstack((top_negative_features, top_positive_features))

    return top_weights, top_features


def get_target_features(estimator, features, top_features, non_zero_features):
    weights = estimator.coef_.toarray().ravel()
    sorted_weight_idcs = np.argsort(weights)
    weights = weights[sorted_weight_idcs]
    features = features[sorted_weight_idcs]

    if non_zero_features:
        weights, features = get_non_zero_features(weights, features)

    if top_features:
        weights, features = get_top_features(weights, features)

    return np.column_stack((weights, features))


def create_top_features_plot(features, result_name):
    features_plot = WeightsFeaturesPlot(
        result_name, weights_features=features, num_top_features=20
    )
    features_plot.plot()
    features_plot.show()


def extract_features(result_path, out_dir, top_features, non_zero):
    result = dumper.load_object(result_path)
    if len(result.feature_extractors) == 0:
        _logger.info("No feature extractors in result file")
        sys.exit(1)

    vectorizer = result.feature_extractors[0]
    features = vectorizer.get_feature_names_out()
    target_features = get_target_features(
        result.estimator, features, top_features, non_zero
    )

    write_to_file(result, target_features, out_dir)

    if top_features:
        create_top_features_plot(target_features, result.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=str, action="store")
    parser.add_argument(
        "--out-dir", "-o", type=str, action="store", help="Path of the output directory"
    )
    parser.add_argument(
        "--top-features",
        action="store",
        help="Extract N top features for each class (0 and 1)",
    )
    parser.add_argument(
        "--non-zero",
        action="store_true",
        help="Extract only features whose weight is not 0.",
    )

    args = parser.parse_args()

    if not args.out_dir:
        args.out_dir = f"{os.getcwd()}/features"

    init_dumper(args.out_dir)
    extract_features(args.result, args.out_dir, args.top_features, args.non_zero)


if __name__ == "__main__":
    main()
