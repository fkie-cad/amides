#!/usr/bin/env python3

import argparse
import sys
import os

from sklearn.svm import SVC as BaseSVC
from sklearnex.svm import SVC as ExtSVC

from amides.persist import Dumper
from amides.data import (
    TrainingResult,
    ValidationResult,
    MultiValidationResult,
    MultiTrainingResult,
)
from amides.utils import set_log_level, get_logger

set_log_level("info")
_logger = get_logger("convert-svc")


dumper = None


def init_dumper(out_dir: str):
    global dumper
    try:
        dumper = Dumper(out_dir)
    except (FileNotFoundError, PermissionError) as err:
        _logger.error(err)
        sys.exit(1)


def create_base_svc_from_ext_svc(ext: ExtSVC) -> BaseSVC:
    base = BaseSVC(
        C=ext.C,
        kernel=ext.kernel,
        degree=ext.degree,
        gamma=ext.gamma,
        coef0=ext.coef0,
        shrinking=ext.shrinking,
        probability=ext.probability,
        tol=ext.tol,
        cache_size=ext.cache_size,
        class_weight=ext.class_weight,
        verbose=ext.verbose,
        max_iter=ext.max_iter,
        decision_function_shape=ext.decision_function_shape,
        break_ties=ext.break_ties,
        random_state=ext.random_state,
    )

    base.support_vectors_ = ext.support_vectors_
    base.n_features_in_ = ext.n_features_in_
    base.fit_status_ = 1
    base.dual_coef_ = ext.dual_coef_
    base.shape_fit_ = ext.shape_fit_
    base.classes_ = ext.classes_
    base.class_weight_ = ext.class_weight_
    base.support_ = ext.support_

    base._intercept_ = ext._intercept_
    base._n_support = ext._n_support
    base._gamma = ext._gamma
    base._probA = ext.probA_
    base._probB = ext.probB_
    base._dual_coef_ = ext._dual_coef_
    base.intercept_ = ext.intercept_
    base.n_iter_ = ext.n_iter_

    return base


def convert_svc_in_single_result(result: TrainingResult) -> TrainingResult:
    if type(result.estimator) == ExtSVC:
        result.estimator = create_base_svc_from_ext_svc(result.estimator)

    return result


def convert_svc_in_multi_result(
    multi_result: MultiTrainingResult,
) -> MultiTrainingResult:
    for result in multi_result.results.values():
        result = convert_svc_in_single_result(result)

    return multi_result


def convert_svc_in_combined_model(model: dict) -> dict:
    model["single"]["clf"] = create_base_svc_from_ext_svc(model["single"]["clf"])

    for rule_model in model["multi"].values():
        rule_model["clf"] = create_base_svc_from_ext_svc(rule_model["clf"])

    return model


def convert_svc(result: str, out_dir: str):
    init_dumper(out_dir)

    loaded = dumper.load_object(result)
    loaded_type = type(loaded)

    if loaded_type in (TrainingResult, ValidationResult):
        converted = convert_svc_in_single_result(loaded)
        dumper.save_object(converted)
    elif loaded_type in (MultiTrainingResult, MultiValidationResult):
        converted = convert_svc_in_multi_result(loaded)
        dumper.save_object(converted)
    elif loaded_type is dict:
        converted = convert_svc_in_combined_model(loaded)
        dumper.save_object(converted, "model")
    else:
        _logger.error(f"Loaded unusable result format: {loaded_type} - Exiting.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        action="store",
        type=str,
        help="Pickled object (.zip) containing sklearnex SVC estimator",
    )
    parser.add_argument(
        "--out-dir", "-o", action="store", type=str, help="Path of the output directory"
    )
    args = parser.parse_args()

    if not args.out_dir:
        args.out_dir = os.getcwd()
        _logger.warning(f"No output directory specified. Using: {args.out_dir}")

    convert_svc(args.result, args.out_dir)


if __name__ == "__main__":
    main()
