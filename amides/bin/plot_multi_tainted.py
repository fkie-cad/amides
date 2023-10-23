#!/usr/bin/env python3
"""This script is used to create a precision-recall plot of multiple evaluation results of models whose training data has been tainted.
The idea of this plot is to visualize the influence of different fractions of tainted training data on the classification performance.
For better comparison, a baseline result (.e.g. with the same parameters, but without tainted data) is provided.
In case of multiple evaluation results (for multiple fractions of tainted data), the script calculates the average precision and 
recall for each fraction of tainting and plots the average values into the figure.
"""

import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from amides.evaluation import BinaryEvaluationResult
from amides.visualization import MultiTaintedPRThresholdsPlot
from amides.persist import get_dumper, PersistError
from amides.utils import get_logger, set_log_level, load_args_from_file

set_log_level("info")
logger = get_logger("plot-multi-tainted")

base_result = None
low_tainted_results = []
medium_tainted_results = []
high_tainted_results = []

out_dir = None
dumper = None
interactive = False

title = None
figure_name = "PRT - Multi-Tainted"


def init_dumper():
    try:
        global dumper
        dumper = get_dumper(out_dir)
    except PersistError as err:
        logger.error(err)
        sys.exit(1)


def load_results(result_paths: list[str]) -> list:
    results = []

    for path in result_paths:
        result = dumper.load_object(path)
        results.append(result)

    return results


def calculate_average_precision_recall(eval_results: BinaryEvaluationResult):
    precision = np.average([result.precision for result in eval_results], axis=0)
    recall = np.average([result.recall for result in eval_results], axis=0)

    average_result = BinaryEvaluationResult(thresholds=eval_results[0].thresholds)
    average_result.precision = precision
    average_result.recall = recall

    return average_result


def create_multi_tainted_prt_plot():
    pr_plot = MultiTaintedPRThresholdsPlot(name=title, timestamp="")
    results = []

    results = load_results([base_result])
    pr_plot.add_evaluation_results("0%", results)

    results = load_results(low_tainted_results)
    results.append(calculate_average_precision_recall(results))
    pr_plot.add_evaluation_results("10%", results)

    results = load_results(medium_tainted_results)
    results.append(calculate_average_precision_recall(results))
    pr_plot.add_evaluation_results("20%", results)

    results = load_results(high_tainted_results)
    results.append(calculate_average_precision_recall(results))
    pr_plot.add_evaluation_results("30%", results)

    pr_plot.plot()

    if interactive:
        plt.plot(interactive=True)

    pr_plot.save(output_path=f"{os.path.join(out_dir, pr_plot.file_name())}.pdf")


def parse_args_and_options(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    if args.config:
        args = load_args_from_file(parser, args.config)
        if not args:
            logger.error("Error loading parameters from file. Exiting")
            sys.exit(1)

    if args.base_result:
        global base_result
        base_result = args.base_result

    if args.low_tainted:
        global low_tainted_results
        low_tainted_results = args.low_tainted

    if args.medium_tainted:
        global medium_tainted_results
        medium_tainted_results = args.medium_tainted

    if args.high_tainted:
        global high_tainted_results
        high_tainted_results = args.high_tainted

    if args.out_dir:
        global out_dir
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(os.getcwd(), "plots")
        logger.warning(
            "No output directory specified. Using current working directory %s",
            out_dir,
        )

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
        "--base-result",
        "-b",
        type=str,
        action="store",
        help="Path of the baseline evaluation result (without tainted benign samples)",
    )
    parser.add_argument(
        "--low-tainted",
        "-lt",
        type=str,
        action="append",
        help="Path of evaluation result(s) with low tainting (10%)",
    )
    parser.add_argument(
        "--medium-tainted",
        "-mt",
        type=str,
        action="append",
        help="Path of evaluation result(s) with medium tainting (20%)",
    )
    parser.add_argument(
        "--high-tainted",
        "-ht",
        type=str,
        action="append",
        help="Path of evaluation result(s) with high tainting (30%)",
    )
    parser.add_argument(
        "--out-dir", "-o", type=str, action="store", help="Path of the output directory"
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

    create_multi_tainted_prt_plot()


if __name__ == "__main__":
    main()
