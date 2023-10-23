#!/usr/bin/env python3
"""This script is used to create plots of evaluation results of misuse classification models. The script takes
precision, recall, f1-score, and mcc-values from the EvaluationResult provided and illustrates them in a
so called precision-recall-thresholds plot. The final plot is saved as .pdf-file into the specified output location.
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from amides.persist import get_dumper, PersistError
from amides.evaluation import BinaryEvaluationResult
from amides.visualization import (
    PrecisionRecallThresholdsPlot,
    MultiPRThresholdsPlot,
)
from amides.utils import get_logger, set_log_level, load_args_from_file

set_log_level("info")
logger = get_logger("plot-pr")

results = []

save = False
output_dir = None
dumper = None
interactive = False

title = None
figure_name = "socbed-rules-tfidf-word"

plot_type = "prt"


def init_dumper():
    try:
        global dumper
        dumper = get_dumper(output_dir)
    except PersistError as err:
        logger.error(err)
        sys.exit(1)


def load_pickled_object(object_path):
    try:
        return dumper.load_object(object_path)
    except (TypeError, PersistError) as err:
        logger.error(err)
        sys.exit(1)


def save_plot(plot):
    try:
        file_path = os.path.join(output_dir, f"{plot.file_name()}.pdf")
        plot.save(file_path)
    except OSError as err:
        logger.error(err)
        sys.exit(1)


def calculate_predict(estimator, scaler, validation_data):
    df_values = estimator.decision_function(validation_data)
    df_scaled = scaler.transform(df_values[:, np.newaxis]).flatten()

    return df_scaled


def add_binary_evaluation_result(result_path, result_name, pr_plot):
    result = load_pickled_object(result_path)
    if not isinstance(result, BinaryEvaluationResult):
        logger.error("Loaded object is no BinaryEvaluationResult")
        return

    pr_plot.add_evaluation_result(result_name, result)


def add_results(pr_plot):
    for name, result_path in results:
        add_binary_evaluation_result(result_path, name, pr_plot)


def create_precision_recall_plot():
    if plot_type == "prt":
        return PrecisionRecallThresholdsPlot(name=title, timestamp="")
    elif plot_type == "multi_prt":
        return MultiPRThresholdsPlot(name=title, timestamp="")


def plot_precision_recall_data():
    pr_plot = create_precision_recall_plot()
    add_results(pr_plot)
    pr_plot.plot()

    if interactive:
        plt.show(block=True)

    if save:
        save_plot(pr_plot)


def parse_args_and_options(parser):
    args = parser.parse_args()

    if args.config:
        args = load_args_from_file(parser, args.config)
        if not args:
            logger.error("Error loading parameters from file. Exiting")
            sys.exit(1)

    if args.result:
        global results
        results = [tuple(result.split(",")) for result in args.result]

    if not results:
        logger.error(
            "Missing path to pickled result object(s): At least one valid object path is required"
        )
        sys.exit(1)

    if args.type:
        global plot_type
        plot_type = args.type

    if args.save:
        global save
        global output_dir

        save = True

        if not args.out_dir:
            output_dir = os.path.join(os.getcwd(), "plots")
            logger.warning(
                "No output directory specified. Using current working directory %s",
                output_dir,
            )
        else:
            output_dir = args.out_dir

        init_dumper()

    if args.interactive:
        global interactive
        interactive = True

    if args.title:
        global title
        title = args.title


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=str,
        action="append",
        help="Path of a pickled EvaluationResult whose data should be visualized",
    )
    parser.add_argument(
        "--type",
        action="store",
        choices=["prt", "multi_prt"],
        default="prt",
        help="Specifiy which type of precision-recall plot should be made",
    )
    parser.add_argument("--save", action="store_true", help="Save plots to file(s)")
    parser.add_argument(
        "--out-dir", type=str, action="store", help="Output directory to save plots"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show plot when computation is finished. Requires manual interaction to close window and finish the script",
    )
    parser.add_argument(
        "--title",
        type=str,
        action="store",
        help="Title of the precision-recall diagram",
    )
    parser.add_argument(
        "--config", type=str, action="store", help="Path to config file."
    )

    parse_args_and_options(parser)
    plot_precision_recall_data()


if __name__ == "__main__":
    main()
