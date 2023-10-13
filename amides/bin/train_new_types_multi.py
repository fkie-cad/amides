#!/usr/bin/env python3

import argparse
import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from amides.data import (
    DataBunch,
    MultiTrainingResult,
    TrainTestValidSplit,
    TrainingResult,
)
from amides.features.normalize import Normalizer
from amides.features.tokenization import AnyWordCharacter
from amides.persist import Dumper
from train import create_symmetric_min_max_scaler


def main():
    args = parse_args()
    multi_train_result = MultiTrainingResult(name=args.result_name)

    num_benign_samples = count_benign_samples(args.benign_samples)
    rules = load_rule_values(args.rule_values)
    model_params = load_model_params(args.model_params)

    for rule in rules:
        rule_values = rule["samples"]
        if not rule_values:
            continue  # skip rules without samples

        other_rule_values = prepare_other_rule_values(rule, rules)

        train_data, vectorizer = prepare_training_data(
            args.benign_samples, num_benign_samples, rule_values, other_rule_values
        )

        estimator, scaler = train_model(train_data, model_params)
        train_result = prepare_result(
            estimator, train_data, vectorizer, scaler, rule["title"]
        )

        multi_train_result.add_result(train_result)

    Dumper(args.out_dir).save_object(multi_train_result)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train classifier for new evasion types."
    )
    parser.add_argument("benign_samples", help="File containing benign samples")
    parser.add_argument("rule_values", help="File containing rule filter values")
    parser.add_argument("model_params", help="File containing model params")
    parser.add_argument("result_name", help="Name of the result file")
    parser.add_argument("out_dir", help="Path of the output directory")
    return parser.parse_args()


def train_samples(benign_tokens, other_rule_tokens, rule_tokens):
    normalizer = Normalizer(max_len_num_values=3)

    for sample in benign_tokens:
        yield sample.rstrip("\n")
    for sample in other_rule_tokens:
        yield normalizer.normalize(sample.rstrip("\n"))
    for sample in rule_tokens:
        yield normalizer.normalize(sample.rstrip("\n"))


def benign_samples(benign_samples_path):
    for sample in open(benign_samples_path, "r", encoding="utf-8"):
        yield sample.rstrip("\n")


def count_benign_samples(path):
    return sum(1 for _ in open(path, "r", encoding="utf-8"))


def load_rule_values(path):
    return [json.loads(rule) for rule in open(path, "r", encoding="utf-8")]


def load_model_params(path):
    with open(path, "r", encoding="utf-8") as model_file:
        return json.load(model_file)


def prepare_other_rule_values(rule, other_rules):
    other_rule_values = []
    for other_rule in other_rules:
        if other_rule["title"] != rule["title"]:
            other_rule_values.extend(other_rule["samples"])

    return other_rule_values


def prepare_training_data(
    benign_samples_path, num_benign_samples, rule_values, other_rule_values
):
    tokenizer = AnyWordCharacter()
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer, analyzer="word", ngram_range=(1, 1)
    )
    vectors = vectorizer.fit_transform(
        train_samples(
            benign_samples(benign_samples_path), other_rule_values, rule_values
        )
    )
    labels = np.array(
        num_benign_samples * [0] + len(other_rule_values) * [0] + len(rule_values) * [1]
    )

    train_data = DataBunch(
        samples=vectors, labels=labels, label_names=["benign", "malicious"]
    )

    return train_data, vectorizer


def train_model(data, model_params):
    estimator = SVC(
        C=model_params["C"],
        kernel=model_params["kernel"],
        class_weight=model_params["class_weight"],
        cache_size=2000,
    )

    estimator.fit(data.samples, data.labels)

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
    )
    train_result.add_feature_extractor(vectorizer)

    return train_result


if __name__ == "__main__":
    main()
