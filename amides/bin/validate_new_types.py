#!/usr/bin/env python3
"""This script validates misuse classification models for other rule and event types.
"""


import sys
import argparse
import numpy as np

from amides.data import DataBunch, ValidationResult
from amides.features.normalize import Normalizer
from amides.persist import Dumper
from amides.utils import load_args_from_file, get_logger, set_log_level


set_log_level("info")
logger = get_logger("validate-new-types")


def main():
    args = parse_args()

    dumper = Dumper(args.out_dir)
    train_result = dumper.load_object(args.result_file)

    validation_data = prepare_validation_data(
        train_result, args.benign_samples, args.evasions_file
    )
    predict = train_result.estimator.decision_function(validation_data.samples)

    validation_result = prepare_validation_result(
        train_result, predict, validation_data
    )

    dumper.save_object(validation_result)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and evaluate a trained classifier."
    )
    parser.add_argument("--result-file", help="zip file containing training results")
    parser.add_argument("--benign-samples", help="File containing benign samples")
    parser.add_argument("--evasions-file", help="File containing evasions")
    parser.add_argument("--out--dir", help="Path of the output directory")
    parser.add_argument("--config", help="Path of config file")

    args = parser.parse_args()

    if args.config:
        args = load_args_from_file(parser, args.config)
    if not args:
        logger.error("Error loading parameters from file. Exiting")
        sys.exit(1)

    return args


def prepare_validation_data(train_result, benign_samples_path, evasions_path):
    num_benign_samples = count_benign_samples(benign_samples_path)
    evasions = load_evasions(evasions_path)

    normalizer = Normalizer(max_len_num_values=3)
    feature_extractor = train_result.feature_extractors[0]
    vectors = feature_extractor.transform(
        validation_samples(normalizer, benign_samples(benign_samples_path), evasions)
    )
    labels = np.array(num_benign_samples * [0] + len(evasions) * [1])
    validation_data = DataBunch(
        samples=vectors, labels=labels, label_names=["benign", "malicious"]
    )

    return validation_data


def benign_samples(benign_samples_path):
    for sample in open(benign_samples_path, "r", encoding="utf-8"):
        yield sample.rstrip("\n")


def validation_samples(normalizer, benign_samples, evasions):
    for sample in benign_samples:
        yield sample
    for sample in evasions:
        yield normalizer.normalize(sample)


def calculate_iteration_values(df_values):
    num_eval_thresholds = 1000
    iter_step = (df_values.max() - df_values.min()) / num_eval_thresholds
    return np.arange(df_values.min(), df_values.max() + iter_step, iter_step)


def load_evasions(evasions_file):
    return [line.rstrip("\n") for line in open(evasions_file, "r", encoding="utf-8")]


def count_benign_samples(path):
    return sum(1 for _ in open(path, "r", encoding="utf-8"))


def prepare_validation_result(train_result, predict, validation_data):
    train_result.data.validation_data = validation_data

    valid_result = ValidationResult(
        estimator=train_result.estimator,
        tainted_share=train_result.tainted_share,
        tainted_seed=train_result.tainted_seed,
        feature_extractors=train_result.feature_extractors,
        scaler=train_result.scaler,
        data=train_result.data,
        predict=predict,
        name=train_result.name,
        timestamp="",
    )

    return valid_result


if __name__ == "__main__":
    main()
