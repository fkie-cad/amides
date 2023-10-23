#!/usr/bin/env python3
"""This script illustrates the evaluation results of the rule attribution model evaluation. Depending
on the specified plot option, this means:
    (1) Distribution of the position of the correct rule evaded in the ranked list of potentially evaded rules
    (2) Cumulative distribution of the position of the correct rule evaded in the ranked list of potentially evaded rules
    (3) Both
"""

import sys
import os
import argparse

from amides.persist import Dumper, PersistError
from amides.evaluation import RuleAttributionEvaluationResult
from amides.visualization import (
    DistributionPlot,
    CumulativeDistributionPlot,
    CombinedDistributionPlot,
)
from amides.utils import (
    get_logger,
    set_log_level,
    load_args_from_file,
)


set_log_level("info")
logger = get_logger("plot-attr")


dumper = None
output_path = None

plot_type = "combined"
title = ""


def init_dumper():
    global dumper

    try:
        if dumper is None:
            dumper = Dumper(output_path)
    except OSError as err:
        logger.error(err)
        sys.exit(1)


def load_object(object_path):
    try:
        return dumper.load_object(object_path)
    except (TypeError, PersistError) as err:
        logger.error(err)
        sys.exit(1)


def save_plot(plot):
    try:
        file_path = os.path.join(output_path, f"{plot.file_name()}.pdf")
        plot.save(file_path)
    except OSError as err:
        logger.error(err)
        sys.exit(1)


def plot_rule_attribution_distribution(eval_rslt_path):
    init_dumper()
    eval_rslt = load_object(eval_rslt_path)

    if type(eval_rslt) is not RuleAttributionEvaluationResult:
        logger.error(
            "Loaded eval result is not of type RuleAttributionEvaluationResult"
        )
        sys.exit(1)
    if plot_type == "dist":
        plot = DistributionPlot(
            x_label="Rules",
            y_label="True Positives (TPs)",
            data=eval_rslt.top_n_hits,
            name=title,
        )
    elif plot_type == "cum_dist":
        plot = CumulativeDistributionPlot(
            x_label="Rules",
            y_label="True Positives (TPs)",
            data=eval_rslt.top_n_hit_rates,
            name=title,
        )
    else:
        plot = CombinedDistributionPlot(data=eval_rslt.top_n_hits, name=title)

    plot.plot()
    plot.save(f"{os.path.join(output_path, plot.file_name())}.pdf")


def parse_args_and_options(parser):
    args = parser.parse_args()

    if args.config:
        args = load_args_from_file(parser, args.config)
        if not args:
            logger.error("Error loading parameters from file. Exiting")
            sys.exit(1)

    if not args.eval_result:
        logger.error("Missing results file. Exiting")
        sys.exit(1)

    if args.plot:
        global plot_type
        plot_type = args.plot

    if args.out_dir:
        global output_path
        output_path = args.out_dir
    else:
        output_dir = os.path.join(os.getcwd(), "plots")
        logger.warning(
            "No output directory specified. Using current working directory %s",
            output_dir,
        )

    if args.title:
        global title
        title = args.title

    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-result",
        type=str,
        action="store",
        help="Path to the RuleAttributionEvaluationResult",
    )
    parser.add_argument(
        "--plot",
        type=str,
        action="store",
        choices=["dist", "cum_dist", "combined"],
        help="Type of plot that should be created",
    )
    parser.add_argument("--out-dir", type=str, action="store", help="Output directory")
    parser.add_argument(
        "--title", type=str, action="store", help="Title of the final figure"
    )
    parser.add_argument(
        "--config", type=str, action="store", help="Path of the config file."
    )
    args = parse_args_and_options(parser)

    plot_rule_attribution_distribution(args.eval_result)


if __name__ == "__main__":
    main()
