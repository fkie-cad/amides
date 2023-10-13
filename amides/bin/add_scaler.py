#!/usr/bin/env python3

import sys
import argparse
import os

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import matthews_corrcoef

from amides.persist import Dumper, PersistError
from amides.data import DataBunch
from amides.features.extraction import CommandlineExtractor
from amides.utils import get_logger, set_log_level, get_current_timestamp
from amides.sigma import RuleSetDataset
from amides.data import (
    TrainingResult,
    MultiTrainingResult,
    ValidationResult,
    MultiValidationResult,
)

set_log_level("info")
logger = get_logger(__name__)

base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../"))
sigma_dir = os.path.join(base_dir, "data/sigma")
events_dir = os.path.join(sigma_dir, "events/windows/process_creation")
rules_dir = os.path.join(sigma_dir, "rules/windows/process_creation")
result_paths = []

out_dir = None

benign_train_samples_path = os.path.join(base_dir, "data/socbed/process_creation/train")
benign_valid_samples_path = os.path.join(base_dir, "data/socbed/process_creation/valid")
num_benign_train_samples = 0
num_benign_valid_samples = 0

dumper = None

malicious_samples = "filter"
optimum_shift = False
mcc_scaling = False
mcc_threshold = 0.1
num_calculation_steps = 50


def init_dumper():
    global dumper

    try:
        if not dumper:
            dumper = Dumper(out_dir)

    except OSError as err:
        logger.err(err)
        sys.exit(1)


def load_pickled_object(path):
    try:
        return dumper.load_object(path.pop(0))
    except (TypeError, PersistError) as err:
        logger.error(err)
        sys.exit(1)
    except IndexError:
        return None


def load_pc_rules_data():
    try:
        return RuleSetDataset(events_dir, rules_dir)
    except FileNotFoundError as err:
        logger.error(err)
        sys.exit(1)


def check_benign_samples(benign_samples_path):
    try:
        return sum(1 for _ in open(benign_samples_path, "rb"))
    except (FileNotFoundError, PermissionError) as err:
        logger.error(err)
        sys.exit(1)


def check_benign_train_samples():
    global num_benign_train_samples

    num_benign_train_samples = check_benign_samples(benign_train_samples_path)


def check_benign_valid_samples():
    global num_benign_valid_samples

    num_benign_valid_samples = check_benign_samples(benign_train_samples_path)


def save_object(obj):
    dumper.save_object(obj)


def create_malicious_sample_list(rule_dataset):
    if malicious_samples == "matches":
        matching_events = rule_dataset.matching_events
        malicious_cmdlines = CommandlineExtractor.extract_commandline(matching_events)
    else:
        malicious_cmdlines = rule_dataset.extract_filter_cmdline_args()

    return malicious_cmdlines


def samples(benign_samples_path, malicious_samples):
    for sample in open(benign_samples_path, "r"):
        yield sample

    for sample in malicious_samples:
        yield sample


def create_labels_array(num_benign_samples, num_malicious_samples):
    labels = []
    labels.extend(num_benign_samples * [0])
    labels.extend(num_malicious_samples * [1])

    return np.array(labels)


def prepare_training_data(extractor, rule_dataset):
    malicious_samples = create_malicious_sample_list(rule_dataset)
    feature_vectors = extractor.transform(
        samples(benign_train_samples_path, malicious_samples)
    )
    labels = create_labels_array(num_benign_valid_samples, len(malicious_samples))

    train_data = DataBunch(
        samples=feature_vectors, labels=labels, label_names=["benign", "matching"]
    )

    return train_data


def prepare_validation_data(extractor, rule_dataset):
    malicious_samples = create_malicious_sample_list(rule_dataset)
    feature_vectors = extractor.transform(
        samples(benign_valid_samples_path, malicious_samples)
    )
    labels = create_labels_array(num_benign_train_samples, len(malicious_samples))

    valid_data = DataBunch(
        samples=feature_vectors, labels=labels, label_names=["benign", "matching"]
    )

    return valid_data


def create_min_max_scaler(df_min, df_max):
    scaler = MinMaxScaler()
    scaler.data_min_ = df_min
    scaler.data_max_ = df_max

    data_range = df_max - df_min
    scaler.data_range_ = data_range

    scale = 1 / data_range
    scaler.scale_ = scale
    scaler.min_ = 0 - df_min * scale

    return scaler


def calculate_shifting_value(mcc, df_iter_values):
    mcc_optimum_idx = np.argmax(mcc)
    optimum_df_value = df_iter_values[mcc_optimum_idx]

    return optimum_df_value


def calculate_iter_values(min_value, max_value):
    iter_step = (max_value - min_value) / num_calculation_steps

    return np.arange(min_value, max_value + iter_step, iter_step)


def calculate_target_df_values(mcc, df_iter_values):
    target_idcs = np.where(mcc > mcc_threshold)[0]

    return df_iter_values[target_idcs]


def calculate_mcc_values(df_values, labels, iter_values):
    mcc = np.zeros(shape=(iter_values.size,))
    for i, threshold in enumerate(iter_values):
        predict = np.where(df_values >= threshold, 1, 0)
        mcc[i] = matthews_corrcoef(y_true=labels, y_pred=predict)

    return mcc


def calculate_symmetric_min_max_df_values(df_values):
    df_min = df_values.min()
    df_max = df_values.max()

    if abs(df_min) > df_max:
        df_max = abs(df_min)
    elif df_max > abs(df_min):
        df_min = df_max * -1.0

    return df_min, df_max


def create_symmetric_mcc_min_max_scaler(df_values, labels):
    df_min, df_max = df_values.min(), df_values.max()
    iter_values = calculate_iter_values(df_min, df_max)
    mcc = calculate_mcc_values(df_values, labels, iter_values)

    target_df_values = calculate_target_df_values(mcc, iter_values)
    target_df_min, target_df_max = calculate_symmetric_min_max_df_values(
        target_df_values
    )

    # Repeat process of MCC optimization after target df-value range was calculated
    # in order to increase precision
    target_iter_values = calculate_iter_values(target_df_min, target_df_max)
    target_mcc = calculate_mcc_values(df_values, labels, target_iter_values)
    target_df_values = calculate_target_df_values(target_mcc, target_iter_values)
    target_df_min, target_df_max = calculate_symmetric_min_max_df_values(
        target_df_values
    )

    return create_min_max_scaler(target_df_min, target_df_max)


def create_symmetric_min_max_scaler(df_values):
    df_min, df_max = calculate_symmetric_min_max_df_values(df_values)

    return create_min_max_scaler(df_min, df_max)


def create_symmetric_scaler(df_values, labels):
    if mcc_scaling:
        return create_symmetric_mcc_min_max_scaler(df_values, labels)
    else:
        return create_symmetric_min_max_scaler(df_values)


def add_scaler_to_multi_train_result(multi_result, pc_rules_data):
    new_multi = MultiTrainingResult(
        name=multi_result.name, timestamp=multi_result.timestamp
    )

    for result in multi_result.results.values():
        rule_dataset = pc_rules_data.get_rule_dataset_by_name(result.name)
        train_result = add_scaler_to_train_result(result, rule_dataset)
        new_multi.add_result(train_result)

    return new_multi


def add_scaler_to_multi_valid_result(multi_result, pc_rules_data):
    new_multi = MultiValidationResult(
        name=multi_result.name, timestamp=multi_result.timestamp
    )

    for result in multi_result.results.values():
        rule_dataset = pc_rules_data.get_rule_dataset_by_name(result.name)
        valid_result = add_scaler_to_valid_result(result, rule_dataset)
        new_multi.add_result(valid_result)

    return new_multi


def add_scaler_to_valid_result(valid_result, rule_dataset):
    if valid_result.data.validation_data is not None:
        data = valid_result.data.validation_data
    else:
        data = prepare_validation_data(valid_result.feature_extractors[0], rule_dataset)

    df_values = valid_result.predict

    scaler = create_symmetric_scaler(df_values, data.labels)

    valid_result = ValidationResult(
        estimator=valid_result.estimator,
        data=valid_result.data,
        scaler=scaler,
        predict=valid_result.predict,
        feature_extractors=valid_result.feature_extractors,
        name=valid_result.name,
        timestamp=get_current_timestamp(),
    )

    return valid_result


def add_scaler_to_train_result(result, rule_dataset):
    if result.data.train_data is not None:
        train_data = result.data.train_data
    else:
        train_data = prepare_training_data(result.feature_extractors[0], rule_dataset)

    df_values = result.estimator.decision_function(train_data.samples)
    scaler = create_symmetric_scaler(df_values, train_data.labels)

    train_result = TrainingResult(
        estimator=result.estimator,
        data=result.data if result.data else None,
        scaler=scaler,
        feature_extractors=result.feature_extractors,
        name=result.name,
        timestamp=result.timestamp,
    )

    return train_result


def add_scaler_to_result(result_path, pc_rules_data):
    result = load_pickled_object(result_path)

    if type(result) is TrainingResult:
        new_result = add_scaler_to_train_result(result, pc_rules_data)
    elif type(result) is ValidationResult:
        new_result = add_scaler_to_valid_result(result, pc_rules_data)
    elif isinstance(result, MultiTrainingResult):
        new_result = add_scaler_to_multi_train_result(result, pc_rules_data)
    elif isinstance(result, MultiValidationResult):
        new_result = add_scaler_to_multi_valid_result(result, pc_rules_data)
    else:
        logger.error("Non-supported result type loaded: %s", type(result))
        sys.exit(1)

    save_object(new_result)


def add_scaler_to_results():
    pc_rules_data = load_pc_rules_data()

    for result_path in result_paths:
        add_scaler_to_result(result_path, pc_rules_data)


def parse_args_and_options(args):
    global result_paths
    result_paths = args.result_path

    if args.mcc_scaling:
        global mcc_scaling
        mcc_scaling = True

    if args.mcc_threshold:
        global mcc_threshold
        mcc_threshold = float(args.mcc_threshold)

    if args.benign_train_samples:
        global benign_train_samples_path
        benign_train_samples_path = args.benign_train_samples
        check_benign_train_samples()

    if args.benign_valid_samples:
        global benign_valid_samples
        benign_valid_samples = args.benign_valid_samples
        check_benign_valid_samples()

    if args.sigma_dir:
        global sigma_dir
        global events_dir
        global rules_dir

        sigma_dir = args.sigma_dir
        events_dir = os.path.join(sigma_dir, "events/windows/process_creation")
        rules_dir = os.path.join(sigma_dir, "rules/windows/process_creation")

    global out_dir
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(os.getcwd(), "models")
        logger.warning("No output dir for results data specified. Using %s", out_dir)

    init_dumper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_path",
        type=str,
        nargs="+",
        action="append",
        help="Path to result instance which is missing scaler-attribute.",
    )
    parser.add_argument(
        "--mcc-scaling",
        action="store_true",
        help="Perform MCC evaluation prior to min-max-scaling.",
    )
    parser.add_argument(
        "--mcc-threshold",
        action="store",
        default=0.1,
        help="Determine the MCC threshold value when performing MCC-based scaling.",
    )
    parser.add_argument(
        "--benign-train-samples",
        type=str,
        action="store",
        help="Path to benign training samples",
    )
    parser.add_argument(
        "--benign-valid-samples",
        type=str,
        action="store",
        help="Path to benign valid samples",
    )
    parser.add_argument(
        "--sigma-dir",
        type=str,
        action="store",
        help="Path to the sigma data containing rule-filters, matches, and evasions.",
    )
    parser.add_argument("--out-dir", type=str, action="store", help="")
    args = parser.parse_args()
    parse_args_and_options(args)

    add_scaler_to_results()


if __name__ == "__main__":
    main()
