#!/usr/bin/env python3
"""This script trains misuse classification models for other event types."""

import sys
import argparse
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from amides.data import DataBunch, TrainTestValidSplit, TrainingResult
from amides.features.normalize import Normalizer
from amides.models.selection import HyperParameterOptimizer
from amides.features.tokenization import AnyWordCharacter
from amides.persist import Dumper
from amides.scale import create_symmetric_min_max_scaler
from amides.utils import get_logger, set_log_level, load_args_from_file


set_log_level("info")
logger = get_logger("train-new-types")


def main():
    args = parse_args()

    train_data, vectorizer = prepare_training_data(
        args.benign_samples, args.rule_values
    )
    estimator, scaler = train_model(train_data)

    train_result = prepare_result(
        estimator, train_data, vectorizer, scaler, args.result_name
    )

    Dumper(args.out_dir).save_object(train_result)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train classifier for new evasion types."
    )
    parser.add_argument(
        "--benign-samples", help="File containing benign training samples"
    )
    parser.add_argument(
        "--rule-values", help="File containing extracted rule filter values"
    )
    parser.add_argument("--result-name", help="Name of the result")
    parser.add_argument("--out-dir", help="Path of the output directory")
    parser.add_argument("--config", help="Path of config file")

    args = parser.parse_args()
    if args.config:
        args = load_args_from_file(parser, args.config)
    if not args:
        logger.error("Error loading parameters from file. Exiting")
        sys.exit(1)

    return args


def prepare_training_data(benign_samples_path, rule_values_path):
    num_benign_samples = count_benign_samples(benign_samples_path)
    rule_values = load_rule_values(rule_values_path)

    normalizer = Normalizer(max_len_num_values=3)

    tokenizer = AnyWordCharacter()
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer, analyzer="word", ngram_range=(1, 1)
    )
    vectors = vectorizer.fit_transform(
        train_samples(normalizer, benign_samples(benign_samples_path), rule_values)
    )

    labels = np.array(num_benign_samples * [0] + len(rule_values) * [1])
    train_data = DataBunch(
        samples=vectors, labels=labels, label_names=["benign", "malicious"]
    )

    return train_data, vectorizer


def train_model(data):
    hp_optimizer = HyperParameterOptimizer(
        SVC(cache_size=2000),
        search_method=GridSearchCV,
        param_grid={
            "kernel": ["linear"],
            "C": np.logspace(-2, 1, num=50),
            "class_weight": ["balanced", None],
        },
        cv_schema=5,
        n_jobs=5,
        scoring=make_scorer(f1_score),
    )
    hp_optimizer.search_best_parameters(data)
    estimator = hp_optimizer.best_estimator

    scaler = create_symmetric_min_max_scaler(estimator.decision_function(data.samples))

    return estimator, scaler


def prepare_result(estimator, data, vectorizer, scaler, result_name):
    train_result = TrainingResult(
        estimator=estimator,
        data=TrainTestValidSplit(train_data=data),
        scaler=scaler,
        tainted_share=0.0,
        tainted_seed=0,
        name=result_name,
        timestamp="",
    )
    train_result.add_feature_extractor(vectorizer)

    return train_result


def train_samples(normalizer, benign_tokens, rule_tokens):
    for sample in benign_tokens:
        yield sample

    for sample in rule_tokens:
        yield normalizer.normalize(sample)


def benign_samples(path):
    for sample in open(path, "r", encoding="utf-8"):
        yield sample.rstrip("\n")


def count_benign_samples(path):
    return sum(1 for _ in open(path, "r", encoding="utf-8"))


def load_rule_values(path):
    return [line.rstrip("\n") for line in open(path, "r", encoding="utf-8")]


if __name__ == "__main__":
    main()
