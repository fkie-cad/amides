#!/usr/bin/env python3

import os
import sys
import argparse

from amides.persist import Dumper, PersistError
from amides.utils import get_logger, set_log_level
from amides.data import TrainingResult


set_log_level("info")
logger = get_logger(__name__)

dumper = None
output_dir = None
combined_name = None

single_path = None
multi_path = None


def init_dumper():
    global dumper

    try:
        if not dumper:
            dumper = Dumper(output_dir)

    except OSError as err:
        logger.err(err)
        sys.exit(1)


def load_result(path):
    try:
        return dumper.load_object(path)
    except (TypeError, PersistError) as err:
        logger.error(err)
        sys.exit(1)


def save_combined_models(combined_models):
    dumper.save_object(combined_models, combined_name)


def get_feature_extractor(result):
    feature_extractor = result.feature_extractors[0]
    feature_extractor.tokenizer = None

    return feature_extractor


def create_clf_vectorizer_dict(result):
    clf_vectorizer_dict = {
        "clf": result.estimator,
        "vectorizer": get_feature_extractor(result),
        "scaler": result.scaler if result.scaler else None,
    }

    return clf_vectorizer_dict


def extract_clf_and_vectorizers(result):
    if isinstance(result, TrainingResult):
        clf_vectorizer_dict = create_clf_vectorizer_dict(result)
    else:
        clf_vectorizer_dict = {}
        for name, result in result.results.items():
            clf_vectorizer_dict[name] = create_clf_vectorizer_dict(result)

    return clf_vectorizer_dict


def combine_models():
    init_dumper()
    single_result = load_result(single_path)
    multi_result = load_result(multi_path)

    combined_models = {
        "single": extract_clf_and_vectorizers(single_result),
        "multi": extract_clf_and_vectorizers(multi_result),
    }

    save_combined_models(combined_models)


def parse_args_and_options(args):
    if args.single:
        global single_path
        single_path = args.single

    if args.multi:
        global multi_path
        multi_path = args.multi

    if single_path is None or multi_path is None:
        logger.error("Missing classifier in order to combine results. Exiting")
        sys.exit(1)

    if args.out_dir:
        global output_dir
        output_dir = args.out_dir
    else:
        output_dir = os.path.join(os.getcwd(), "combined")
        logger.warning(
            "No output directory for combined results specified. Using %s", output_dir
        )

    if args.name:
        global combined_name
        combined_name = args.name
    else:
        logger.error("No ouput name given. Exiting")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single",
        type=str,
        action="store",
        help="Path to results file (TrainingResult, ValidationResult) containing the single model",
    )
    parser.add_argument(
        "--multi",
        type=str,
        action="store",
        help="Path to results file (ValidationResults) holding the multi model",
    )
    parser.add_argument(
        "--out-dir", type=str, action="store", help="Specify output directory"
    )
    parser.add_argument(
        "--name",
        type=str,
        action="store",
        help="Specify name of the combined output file",
    )

    args = parser.parse_args()
    parse_args_and_options(args)

    combine_models()


if __name__ == "__main__":
    main()
