#!/usr/bin/env python3

import sys
import os
import argparse
import json
import numpy as np

from typing import Iterable

from amides.data import TrainingResult
from amides.utils import set_log_level, get_logger
from amides.persist import get_dumper
from amides.features.normalize import Normalizer
from amides.sigma import RuleSetDataset
from amides.features.extraction import CommandlineExtractor
from amides.events import Events


set_log_level("info")
_logger = get_logger("confidence")


base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../"))
sigma_dir = os.path.join(base_dir, "data/sigma")
events_dir = os.path.join(sigma_dir, "events/windows/process_creation")
rules_dir = os.path.join(sigma_dir, "rules/windows/process_creation")

benign_samples = os.path.join(base_dir, "data/socbed/cmdlines")
result = None

dumper = None
decision_threshold = None
samples = "rule_filter"
collect_features_and_weights = False


def init_dumper():
    global dumper

    dumper = get_dumper(out_dir)


def load_result() -> TrainingResult:
    return dumper.load_object(result)


def write_to_file(output: dict, file_name: str):
    out_file_path = os.path.join(out_dir, file_name)

    with open(out_file_path, "w", encoding="utf-8") as out_file:
        json.dump(output, out_file, indent=4)


def write_cf_results_to_file(cf_values: list[dict], file_name: str):
    output = {"cf_values": cf_values, "num_elements": len(cf_values)}
    write_to_file(output, file_name)


def write_features_weights_to_file(features_weights: list[tuple], file_name: str):
    output = {"features_weights": features_weights}
    write_to_file(output, file_name)


def build_file_name(r_type: str, result_name: str) -> str:
    return f"cfn_{samples}_{r_type}_{result_name}.json"


def benign_samples_iterator() -> str:
    with open(benign_samples, "r", encoding="utf-8") as in_file:
        for line in in_file:
            yield json.loads(line.rstrip("\n"))


def rule_filter_iterator() -> str:
    rule_set_data = RuleSetDataset(events_dir, rules_dir)
    rule_filters = rule_set_data.extract_field_values_from_filter(
        search_fields=["process.command_line"]
    )

    for rule_filter in rule_filters:
        yield rule_filter


def events_iterator(events: Events) -> str:
    cmdlines = CommandlineExtractor.extract_commandline(events.data)

    for cmdline in cmdlines:
        yield cmdline


def add_samples_to_values(results: dict[str, dict]) -> dict[str, dict]:
    for sample, result in results.items():
        result["sample"] = sample


def create_list_sorted_by_df_values(results: dict[str, dict]) -> list[dict]:
    return sorted(results, key=lambda x: x["df_value"], reverse=True)


def extract_estimator_vectorizer_scaler(result: TrainingResult) -> tuple[any, any]:
    try:
        estimator = result.estimator
        vectorizer = result.feature_extractors[0]
        scaler = result.scaler

        return estimator, vectorizer, scaler
    except KeyError as err:
        _logger.error(err)
        sys.exit(1)


def get_sorted_features_and_weights(weights, feature_names, target_feature_names):
    target_feature_name_idcs = np.where(np.in1d(feature_names, target_feature_names))[0]
    target_weights = weights[target_feature_name_idcs]
    target_feature_names = feature_names[target_feature_name_idcs]
    sorted_weights_idcs = np.argsort(target_weights)[::-1]

    return list(
        zip(
            target_feature_names[sorted_weights_idcs],
            target_weights[sorted_weights_idcs],
        )
    )


def calculate_contributions_and_feature_names(
    sample: np.ndarray, weights: np.ndarray, feature_names: list
) -> tuple[np.ndarray, np.ndarray]:
    sample = convert_and_reshape(sample)
    contributions = np.multiply(sample, weights)
    non_zero_idcs = np.nonzero(contributions)

    target_contributions = contributions[non_zero_idcs]
    target_feature_names = feature_names[non_zero_idcs]

    return list(target_contributions), list(target_feature_names)


def convert_and_reshape(csr_matrix):
    arr = csr_matrix.toarray()

    return arr.reshape((arr.size,))


def calculate_confidence_values(result: TrainingResult, samples: Iterable[str]) -> dict:
    estimator, vectorizer, scaler = extract_estimator_vectorizer_scaler(result)
    all_feature_names = vectorizer.get_feature_names_out()
    weights = convert_and_reshape(estimator.coef_)
    total_features = set()

    normalizer = Normalizer()

    results = {}

    for sample in samples:
        normalized = normalizer.normalize(sample)
        transformed = vectorizer.transform([normalized])
        non_zero = transformed.count_nonzero()
        contributions, feature_names = calculate_contributions_and_feature_names(
            transformed, weights, all_feature_names
        )
        total_features.update(feature_names)

        try:
            df_value = estimator.decision_function(transformed)[0]
            confidence = scaler.transform(df_value.reshape(-1, 1)).flatten()[0]
        except MemoryError:
            df_value = 0.0
            confidence = 0.0

        if sample not in results:
            results[sample] = {
                "sample": sample,
                "normalized": normalized,
                "feature_names": feature_names,
                "num_non_zero_entries": non_zero,
                "df_value": df_value,
                "confidence": confidence,
                "contributions": contributions,
            }

    if collect_features_and_weights:
        feature_weights = get_sorted_features_and_weights(
            weights, all_feature_names, list(total_features)
        )
    else:
        feature_weights = None

    return results, feature_weights


def split_according_decision_threshold(cf_values: list[dict]):
    malicious, benign = [], []

    for values in cf_values.values():
        if values["df_value"] > 0.0:
            malicious.append(values)
        else:
            benign.append(values)

    return malicious, benign


def calculate_benign_samples_confidence_values(result: TrainingResult) -> dict:
    return calculate_confidence_values(result, benign_samples_iterator())


def calculate_rule_filter_confidence_values(result: TrainingResult) -> dict:
    return calculate_confidence_values(result, rule_filter_iterator())


def calculate_evasion_confidence_values(result: TrainingResult) -> dict:
    rule_set_data = RuleSetDataset(events_dir, rules_dir)
    evasions = rule_set_data.evasions

    return calculate_confidence_values(result, events_iterator(evasions))


def calculate_matches_confidence_values(result: TrainingResult) -> dict:
    rule_set_data = RuleSetDataset(events_dir, rules_dir)
    matches = rule_set_data.matches

    return calculate_confidence_values(result, events_iterator(matches))


def create_normalization_confidence_values_list():
    init_dumper()
    result = load_result()

    if samples == "benign":
        cf_values, features_weights = calculate_benign_samples_confidence_values(result)
    elif samples == "rule_filter":
        cf_values, features_weights = calculate_rule_filter_confidence_values(result)
    elif samples == "evasions":
        cf_values, features_weights = calculate_evasion_confidence_values(result)
    elif samples == "matches":
        cf_values, features_weights = calculate_matches_confidence_values(result)

    result_file_name = result.file_name()

    if decision_threshold is not None:
        malicious_cf_values, benign_cf_values = split_according_decision_threshold(
            cf_values
        )

        if samples in ("matches", "evasions", "rule_filter"):
            malicious_file_name = build_file_name("tp", result_file_name)
            benign_file_name = build_file_name("fn", result_file_name)
            malicious_cf_values = create_list_sorted_by_df_values(malicious_cf_values)
            benign_cf_values = create_list_sorted_by_df_values(benign_cf_values)
            write_cf_results_to_file(malicious_cf_values, malicious_file_name)
            write_cf_results_to_file(benign_cf_values, benign_file_name)
        else:
            malicious_file_name = build_file_name("fp", result_file_name)
            malicious_cf_values = create_list_sorted_by_df_values(malicious_cf_values)
            write_cf_results_to_file(malicious_cf_values, malicious_file_name)

    else:
        file_name = build_file_name("all", result_file_name)
        cf_values = create_list_sorted_by_df_values(cf_values)
        write_cf_results_to_file(cf_values, file_name)

    if features_weights is not None:
        features_weights_file_name = build_file_name(
            "features_weights", result_file_name
        )
        write_features_weights_to_file(features_weights, features_weights_file_name)


def parse_args_and_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", action="store")
    parser.add_argument(
        "--samples",
        "-s",
        action="store",
        type=str,
        choices=["benign", "rule_filter", "evasions", "matches"],
    )
    parser.add_argument(
        "--benign-samples",
        action="store",
        type=str,
        help="Path of a benign samples file",
    )
    parser.add_argument(
        "--sigma-dir", action="store", type=str, help="Path of the sigma data directory"
    )
    parser.add_argument(
        "--decision-threshold",
        action="store",
        type=float,
        help="Sort samples into TP/FN (TN/FP) according to the specified decision threshold",
    )
    parser.add_argument(
        "--collect-features-weights",
        action="store_true",
        help="Collect used feature names and their corresponding weights",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        action="store",
        type=str,
        help="The output directory",
    )
    args = parser.parse_args()

    global result
    result = args.result

    if args.samples:
        global samples
        samples = args.samples

    if args.benign_samples:
        global benign_samples
        benign_samples = args.benign_samples

    if samples == "benign" and not benign_samples:
        _logger.error(f"Missing benign samples file for 'benign' samples. Exiting")
        sys.exit(1)

    if args.collect_features_weights:
        global collect_features_and_weights
        collect_features_and_weights = True

    global out_dir
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.getcwd()
        _logger.warning(f"No output directory specified. Using {out_dir}")

    if args.sigma_dir:
        global sigma_dir
        sigma_dir = args.sigma_dir

    if args.decision_threshold:
        global decision_threshold
        decision_threshold = args.decision_threshold


def main():
    parse_args_and_options()
    create_normalization_confidence_values_list()


if __name__ == "__main__":
    main()
