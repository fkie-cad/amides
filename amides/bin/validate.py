#!/usr/bin/env python3
"""This script is used to validate models trained for the AMIDES misuse classification and rule attribution components
using a set of benign samples and Sigma rule evasions. Benign samples are provided in the same format as for train.py.

After loading estimator and feature extractor from the given TrainingResult, the feature extractor
transforms the given benign validation samples and Sigma evasions into feature vectors. Afterwards, the model is used
to calculate decision function values for the transformed validation samples. In case of a MultiTrainingResult, the
step is repeated for each rule model provided.

The calculated decision function values and feature vectors are stored together with the rest of the TrainingResult into
a ValidationResult, which is then pickled. In case of a rule attribution model validation, ValidationResult objects for 
each rule models are pickled into a single MultiValidationResult object.
"""

import sys
import os
import argparse
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler

from amides.utils import (
    get_logger,
    set_log_level,
    execution_time,
    load_args_from_file,
)
from amides.persist import Dumper, PersistError
from amides.features.extraction import CommandlineExtractor
from amides.features.normalize import normalize
from amides.evaluation import BinaryEvaluationResult
from amides.data import (
    DataBunch,
    MultiTrainingResult,
    MultiValidationResult,
    TrainingResult,
    ValidationResult,
)
from amides.sigma import RuleSetDataset, RuleDataset

set_log_level("info")
_logger = get_logger("validate")

base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../"))
sigma_dir = os.path.join(base_dir, "data/sigma")
events_dir = os.path.join(sigma_dir, "events/windows/process_creation")
rules_dir = os.path.join(sigma_dir, "rules/windows/process_creation")

rule_set_data = None
result_paths = None

benign_samples_path = os.path.join(base_dir, "data/socbed/process_creation/valid")
num_benign_samples = 0

save_data = False
save_vectorizer = False

malicious_samples_type = "evasions"

adapt_scaling = False

evaluate = False
num_eval_thresholds = None

result_name = None

output_dir = None
dumper = None


def init_dumper():
    global dumper

    try:
        if dumper is None:
            dumper = Dumper(output_dir)
    except OSError as err:
        _logger.error(err)
        sys.exit(1)


def load_result(result_path: str):
    try:
        return dumper.load_object(result_path)
    except (TypeError, PersistError) as err:
        _logger.error(err)
        sys.exit(1)


def load_pc_rules_dataset():
    global rule_set_data

    try:
        if not rule_set_data:
            rule_set_data = RuleSetDataset()
            rule_set_data.load_rule_set_data(events_dir, rules_dir)

        return rule_set_data
    except FileNotFoundError as err:
        _logger.error(err)
        sys.exit(1)


def save_result(result: ValidationResult):
    dumper.save_object(result)


def check_benign_valid_samples():
    global num_benign_samples

    try:
        num_benign_samples = sum(1 for _ in open(benign_samples_path, "rb"))
    except (FileNotFoundError, PermissionError) as err:
        _logger.error(err)
        sys.exit(1)


def get_feature_extractor(result: TrainingResult):
    if not result.feature_extractors:
        return None

    return result.feature_extractors[0]


def get_validation_data(result: TrainingResult):
    if type(result) is ValidationResult:
        if result.data.validation_data is not None:
            return result.data.validation_data

    return None


def calculate_df_values(result: TrainingResult, samples: np.ndarray):
    if type(result) is ValidationResult:
        if result.predict is not None:
            df_values = result.predict
    else:
        df_values = calculate_predict(result.estimator, samples)

    return df_values


def create_labels_array(num_malicious_samples: int):
    labels = []
    labels.extend((num_benign_samples) * [0])
    labels.extend(num_malicious_samples * [1])

    return np.array(labels)


def create_malicious_sample_list(
    rule_dataset: RuleDataset, tainted_share: float, tainted_seed: int
):
    if malicious_samples_type == "evasions":
        events = rule_dataset.evasions
    else:
        events = rule_dataset.matches

    if tainted_share > 0.0:
        _, events = events.create_random_split(
            [tainted_share, 1.0 - tainted_share], seed=tainted_seed
        )

    samples = CommandlineExtractor.extract_commandline(events.data)

    return normalize(samples)


def validation_samples(malicious_samples: list[str]):
    for sample in open(benign_samples_path, "r", encoding="utf-8"):
        yield sample.rstrip("\n")

    for sample in malicious_samples:
        yield sample


@execution_time
def prepare_validation_data(result: TrainingResult, rule_dataset: RuleDataset):
    feature_extractor = get_feature_extractor(result)
    if not feature_extractor:
        _logger.error("Result does not contain required feature extractors. Exiting")
        return None

    malicious_samples = create_malicious_sample_list(
        rule_dataset, result.tainted_share, result.tainted_seed
    )

    feature_vectors = feature_extractor.transform(validation_samples(malicious_samples))
    labels = create_labels_array(len(malicious_samples))

    validation_data = DataBunch(
        samples=feature_vectors, labels=labels, label_names=["benign", "malicious"]
    )

    return validation_data


def scale_df_values(df_values: np.ndarray, scaler: MinMaxScaler):
    df_scaled = scaler.transform(df_values[:, np.newaxis]).flatten()

    return df_scaled


@execution_time
def calculate_predict(estimator: BaseEstimator, valid_data: DataBunch):
    df_values = estimator.decision_function(valid_data)

    return df_values


def prepare_validation_result(
    result: TrainingResult, predict: np.ndarray, validation_data: DataBunch
):
    if save_data:
        result.data.validation_data = validation_data

    valid_result = ValidationResult(
        estimator=result.estimator,
        tainted_share=result.tainted_share,
        tainted_seed=result.tainted_seed,
        feature_extractors=result.feature_extractors if save_vectorizer else None,
        scaler=result.scaler,
        data=result.data,
        predict=predict,
        name=result_name if result_name else result.name,
        timestamp="",
    )

    return valid_result


def validate_multi_model(multi_result: MultiTrainingResult):
    _logger.info("Validating model %s", multi_result.name)
    check_benign_valid_samples()
    pc_rules_data = load_pc_rules_dataset()
    multi_valid_result = MultiValidationResult(name=multi_result.name)

    for result in multi_result.results.values():
        _logger.info("Validating model %s", result.name)

        rule_dataset = pc_rules_data.get_rule_dataset_by_name(result.name)
        validation_data = prepare_validation_data(result, rule_dataset)
        predict = calculate_predict(result.estimator, validation_data.samples)

        valid_result = prepare_validation_result(result, predict, validation_data)
        multi_valid_result.add_result(valid_result)

    save_result(multi_valid_result)


def evaluate_single_model(valid_result: ValidationResult):
    _logger.info("Evaluating model %s", valid_result.name)

    validation_data = get_validation_data(valid_result)
    if validation_data is None:
        check_benign_valid_samples()
        pc_rules_dataset = load_pc_rules_dataset()
        validation_data = prepare_validation_data(valid_result, pc_rules_dataset)

    df_values = calculate_df_values(valid_result, validation_data.samples)

    scaled_df_values = scale_df_values(df_values, valid_result.scaler)
    iter_step = 1 / num_eval_thresholds
    iter_values = np.arange(0, 1 + iter_step, iter_step)

    # iter_step = (df_values.max() - df_values.min()) / num_eval_thresholds
    # iter_values = np.arange(df_values.min(), df_values.max() + iter_step, iter_step)

    eval_result = BinaryEvaluationResult(
        name=valid_result.name,
        thresholds=iter_values,
        timestamp=valid_result.timestamp,
    )

    eval_result.evaluate(validation_data.labels, scaled_df_values)

    save_result(eval_result)


def validate_single_model(result: TrainingResult):
    _logger.info("Validating model %s", result.name)

    validation_data = get_validation_data(result)
    if validation_data is None:
        check_benign_valid_samples()
        pc_rules_data = load_pc_rules_dataset()
        validation_data = prepare_validation_data(result, pc_rules_data)

    predict = calculate_predict(result.estimator, validation_data.samples)

    valid_result = prepare_validation_result(result, predict, validation_data)

    save_result(valid_result)

    return valid_result


def validate_model(result_path: str):
    result = load_result(result_path)

    if type(result) is TrainingResult:
        valid_result = validate_single_model(result)
        if evaluate:
            evaluate_single_model(valid_result)
    elif type(result) is ValidationResult:
        if evaluate:
            evaluate_single_model(result)
    elif type(result) is MultiTrainingResult:
        valid_result = validate_multi_model(result)
    else:
        _logger.error("Loaded object is not of supported result type. Exiting.")
        sys.exit(1)


def validate_models():
    init_dumper()

    for result_path in result_paths:
        try:
            validate_model(result_path)
        except FileNotFoundError as err:
            _logger.error(err)
            continue


def parse_args_and_options(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    if args.config:
        args = load_args_from_file(parser, args.config)
        if not args:
            _logger.error("Error loading parameters from file. Exiting")
            sys.exit(1)

    if args.result_path:
        global result_paths
        result_paths = args.result_path

    if not result_paths:
        _logger.error("No result file specified. Exiting.")
        sys.exit(1)

    if args.events_dir:
        global events_dir
        events_dir = args.events_dir

    if args.rules_dir:
        global rules_dir
        rules_dir = args.rules_dir

    if args.benign_samples:
        global benign_samples_path
        benign_samples_path = args.benign_samples

    if args.malicious_samples_type:
        global malicious_samples_type
        malicious_samples_type = args.malicious_samples_type

    if args.save_data:
        global save_data
        save_data = True

    if args.save_vectorizer:
        global save_vectorizer
        save_vectorizer = True

    if args.evaluate:
        global evaluate
        evaluate = True

    if args.num_eval_thresholds:
        global num_eval_thresholds
        num_eval_thresholds = args.num_eval_thresholds

    if args.zero_to_zero:
        global zero_to_zero
        zero_to_zero = True

    global output_dir
    if args.out_dir:
        output_dir = args.out_dir
    else:
        output_dir = os.path.join(os.getcwd(), "models")
        _logger.warning(
            "No output directory specified. Using current working directory %s",
            output_dir,
        )

    if args.result_name:
        global result_name
        result_name = args.result_name


def main():
    parser = argparse.ArgumentParser(
        description="Validate misuse and rule attribution models for AMIDES"
    )
    parser.add_argument(
        "--result-path",
        type=str,
        action="append",
        help="Path of a pickled TrainingResult or MultiTrainingResult",
    )
    parser.add_argument(
        "--benign-samples",
        type=str,
        action="store",
        help="Path of the benign validation samples file (.txt)",
    )
    parser.add_argument(
        "--events-dir",
        type=str,
        nargs="?",
        action="store",
        help="Path of the directory with Sigma rule matches and evasions (.json)",
    )
    parser.add_argument(
        "--rules-dir",
        type=str,
        nargs="?",
        action="store",
        help="Path of the directory with Sigma detection rules (.yml)",
    )
    parser.add_argument(
        "--malicious-samples-type",
        type=str,
        action="store",
        choices=["matches", "evasions"],
        help="Type of malicious samples used for validation",
    )
    parser.add_argument(
        "--save-vectorizer",
        action="store_true",
        help="Save vectorizer used for sample transformation (if provided by TrainingResult)",
    )
    parser.add_argument(
        "--adapt-scaling",
        action="store_true",
        help="Adapt given scaler to symmetric MCC-scaler using invsere scale transformation on the validation data",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the provided model(s) using validation data and scaled model decision function values.",
    )
    parser.add_argument(
        "--num-eval-thresholds",
        type=int,
        action="store",
        default=50,
        help="Number of evaluation thresholds used when model(s) are evaluated",
    )
    parser.add_argument(
        "--zero-to-zero",
        action="store_true",
        help="Set decision function values of all-zero feature vectors to 0.0",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        action="store",
        help="Output directory to save result files",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Specify if validation data should be added to ValidationResult",
    )
    parser.add_argument(
        "--result-name", type=str, action="store", help="Specifies the result's name"
    )
    parser.add_argument(
        "--config", type=str, action="store", help="Path to config file"
    )

    parse_args_and_options(parser)

    validate_models()


if __name__ == "__main__":
    main()
