#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import itertools

from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

from amides.persist import Dumper
from amides.data import ValidationResult, TrainingResult
from amides.utils import get_logger, set_log_level, get_current_timestamp
from amides.evaluation import BinaryEvaluationResult
from amides.utils import load_args_from_file


set_log_level("info")
_logger = get_logger("evaluate-mcc-scaling")


valid_results = []
train_results = []
mcc_values = []

zero_null_vectors = False
mcc_threshold = 0.1

dumper = None
save = False


num_evaluation_thresholds = 50


def init_dumper(out_dir):
    global dumper

    try:
        if not dumper:
            dumper = Dumper(out_dir)
    except OSError as err:
        _logger.error(err)
        sys.exit(1)


def load_result(result_path):
    return dumper.load_object(result_path)


def load_valid_result(result_path):
    valid_result = load_result(result_path)

    df_values = valid_result.predict
    labels = valid_result.data.validation_data.labels
    result_name = valid_result.name

    return df_values, labels, result_name


def load_mcc_values(mcc_values_path):
    if mcc_values_path:
        mcc_values = load_result(mcc_values_path)
        return mcc_values["mcc"]

    return None


def save_mcc_values(mcc_values, df_iter_values, result_name, timestamp):
    result = {
        "mcc": mcc_values,
        "df_iter_values": df_iter_values,
    }
    dumper.save_object(result, f"mcc_{result_name}_{timestamp}")


def save_eval_result(result):
    dumper.save_object(result)


def save_valid_result(valid_result, scaler, timestamp):
    valid_result = ValidationResult(
        estimator=valid_result.estimator,
        predict=valid_result.predict,
        scaler=scaler,
        data=valid_result.data,
        feature_extractors=valid_result.feature_extractors,
        name=valid_result.name,
        timestamp=timestamp,
    )

    dumper.save_object(valid_result)


def evaluate_scaling(valid_result, scaled_df_values):
    iter_step = 1 / num_evaluation_thresholds
    eval_result = BinaryEvaluationResult(
        name=valid_result.name,
        thresholds=np.arange(
            0,
            1 + iter_step,
            iter_step,
        ),
        timestamp=get_current_timestamp(),
    )
    validation_data = valid_result.data.validation_data
    eval_result.evaluate(validation_data.labels, scaled_df_values)

    save_eval_result(eval_result)


def calculate_iter_values(min_value, max_value):
    df_iter = (max_value - min_value) / num_evaluation_thresholds

    iter_values = np.arange(
        min_value,
        max_value + df_iter,
        df_iter,
    )

    return iter_values


def calculate_mcc(df_values, labels, iter_values):
    mcc = np.zeros(shape=(len(iter_values),))

    for i, threshold in enumerate(iter_values):
        _logger.debug("MCC Iteration: %d", i)
        predict = np.where(df_values >= threshold, 1, 0)
        mcc[i] = matthews_corrcoef(y_true=labels, y_pred=predict)

    return mcc


def calculate_precision(df_values, labels, iter_values):
    precision = np.zeros(shape=(len(iter_values),))

    for i, threshold in enumerate(iter_values):
        _logger.debug("Precision Iteration: %d", i)
        predict = np.where(df_values >= threshold, 1, 0)
        precision[i] = precision_score(y_true=labels, y_pred=predict, zero_division=1)

    return precision


def calculate_recall(df_values, labels, iter_values):
    recall = np.zeros(shape=(len(iter_values),))

    for i, threshold in enumerate(iter_values):
        _logger.debug("Recall Iteration: %d", i)
        predict = np.where(df_values >= threshold, 1, 0)
        recall[i] = recall_score(y_true=labels, y_pred=predict, zero_division=1)

    return recall


def calculate_mcc_ranges(mcc):
    for i in np.arange(0, 1, 0.1):
        mcc_below_threshold = mcc[mcc < i].size
        relative_mcc_below_threshold = mcc_below_threshold / mcc.size * 100.0
        _logger.debug(
            "Df-values with MCC lower than %0.2f (relative): %d (%0.2f%%)",
            i,
            mcc_below_threshold,
            relative_mcc_below_threshold,
        )


def create_min_max_scaler(min_df_value, max_df_value):
    scaler = MinMaxScaler()
    scaler.data_min_ = min_df_value
    scaler.data_max_ = max_df_value

    data_range = max_df_value - min_df_value
    scaler.data_range_ = data_range

    scale = 1 / data_range
    scaler.scale_ = scale
    scaler.min_ = 0 - min_df_value * scale

    return scaler


def calculate_shifting_value(mcc, df_iter_values):
    mcc_optimum_idx = np.argmax(mcc)
    optimum_df_value = df_iter_values[mcc_optimum_idx]

    return optimum_df_value


def get_target_df_min_max_values(mcc, iter_values):
    target_idcs = np.where(mcc > mcc_threshold)[0]
    target_iter_values = iter_values[target_idcs]
    target_df_min, target_df_max = (
        target_iter_values.min(),
        target_iter_values.max(),
    )

    if abs(target_df_min) > target_df_max:
        target_df_max = abs(target_df_min)
    elif target_df_max > abs(target_df_min):
        target_df_min = target_df_max * -1.0

    return target_df_min, target_df_max


def adjust_null_vector_output_labels(
    vectors: np.ndarray, labels: np.ndarray, df_values: np.ndarray
) -> np.ndarray:
    zero_vec_indices = np.where((vectors.sum(axis=1) == 0))[0]
    df_min = df_values.min()
    labels[zero_vec_indices] = 0
    df_values[zero_vec_indices] = df_min

    return labels


def prepare_values_and_labels(
    valid_result: ValidationResult, train_result: TrainingResult | None
) -> tuple[np.ndarray, np.ndarray]:
    if train_result:
        df_values = train_result.estimator.decision_function(
            train_result.data.train_data.samples
        )
        labels = train_result.data.train_data.labels

        if zero_null_vectors:
            labels = adjust_null_vector_output_labels(
                train_result.data.train_data.samples, labels, df_values
            )
    else:
        df_values = valid_result.predict
        labels = valid_result.data.validation_data.labels

        if zero_null_vectors:
            labels = adjust_null_vector_output_labels(
                valid_result.data.validation_data.samples, labels, df_values
            )

    return df_values, labels


def evaluate_mcc_scaling_with_optimum_shift():
    result_paths = list(itertools.zip_longest(valid_results, train_results, mcc_values))

    for valid_rslt_path, train_rslt_path, mcc_val_path in result_paths:
        # timestamp = get_current_timestamp()
        # time.sleep(5)
        valid_result = load_result(valid_rslt_path)
        train_result = load_result(train_rslt_path) if train_rslt_path else None

        df_values, labels = prepare_values_and_labels(valid_result, train_result)

        df_iter_values = calculate_iter_values(df_values.min(), df_values.max())
        mcc = calculate_mcc(df_values, labels, df_iter_values)

        # Get (df_min, df_max)-range where MCC > mcc_threshold
        target_df_min, target_df_max = get_target_df_min_max_values(mcc, df_iter_values)
        target_iter_values = calculate_iter_values(target_df_min, target_df_max)

        # Calculate MCC values for (df_min, df_max)
        target_mcc = load_mcc_values(mcc_val_path)
        if target_mcc is None:
            target_mcc = calculate_mcc(df_values, labels, target_iter_values)
            # save_mcc_values(
            #    target_mcc, target_iter_values, valid_result.name, timestamp
            # )

        # Calculate shifting value so that MCC optimum is at df-value 0
        shifting_value = calculate_shifting_value(target_mcc, target_iter_values)
        # Shift df values accordingly

        if train_result:
            df_values = valid_result.predict
            labels = valid_result.data.validation_data.labels

        shifted_df_values = df_values - shifting_value

        # Create symmetric scaler for the shifted df values for (df_min, df_max)
        scaler = create_min_max_scaler(target_df_min, target_df_max)
        scaled_df_values = scaler.transform(shifted_df_values[:, np.newaxis]).flatten()

        iter_values = calculate_iter_values(0, 1)

        eval_result = BinaryEvaluationResult(
            thresholds=iter_values,
            name=valid_result.name,
            timestamp="",
        )
        eval_result.evaluate(labels, scaled_df_values)

        save_eval_result(eval_result)


def parse_args_and_options(parser):
    args = parser.parse_args()

    if args.config:
        args = load_args_from_file(parser, args.config)
        if not args:
            _logger.error("Error loading parameters from file. Exiting")
            sys.exit(1)

    if not args.valid_results:
        _logger.error("No valid results specified. Exiting.")
        sys.exit(1)

    global valid_results
    valid_results = args.valid_results

    if args.num_eval_thresholds:
        global num_evaluation_thresholds
        num_evaluation_thresholds = args.num_eval_thresholds

    if args.train_results:
        global train_results
        train_results = args.train_results

    if args.mcc_values:
        global mcc_values
        mcc_values = args.mcc_values

    if args.zero_null_vecs:
        global zero_null_vectors
        zero_null_vectors = args.zero_null_vecs

    if not args.out_dir:
        args.out_dir = f"{os.getcwd()}/models"

    init_dumper(args.out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--valid-results",
        type=str,
        action="append",
        default=[],
        help="Validation results that require mcc-scaling",
    )
    parser.add_argument(
        "--train-results",
        type=str,
        action="append",
        default=[],
        help="Training result that requires mcc-sclaing",
    )
    parser.add_argument(
        "--mcc-values",
        type=str,
        action="append",
        default=[],
        help="Path of the associated MCC values (if available)",
    )
    parser.add_argument(
        "--num-eval-thresholds",
        action="store",
        default=50,
        help="Specifiy the number of evaluation thresholds",
    )
    parser.add_argument(
        "--zero-null-vecs",
        action="store_true",
        help="Set all all-zeor-vectors to zero output",
    )
    parser.add_argument(
        "--out-dir", type=str, action="store", help="Path of the output directory"
    )
    parser.add_argument(
        "--config", type=str, action="store", help="Path to config file."
    )

    parse_args_and_options(parser)

    evaluate_mcc_scaling_with_optimum_shift()


if __name__ == "__main__":
    main()
