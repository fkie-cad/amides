import argparse
import sys
import os

import numpy as np

from amides.visualization import SwarmPlot, BoxPlot, ViolinPlot
from amides.data import (
    ValidationResult,
)
from amides.persist import Dumper
from amides.utils import get_logger, set_log_level

set_log_level("info")
logger = get_logger(__name__)


file_name = None
result_path = None

plot_type = "box"
data = []

scale_df = False


def load_result(result_path):
    try:
        dumper = Dumper()
        return dumper.load_object(result_path)
    except (FileNotFoundError, PermissionError) as err:
        logger.error(err)
        sys.exit(1)


def get_y_label():
    return "Scaled decision function values" if scale_df else "Decision function values"


def create_box_plot(df_values, name):
    plot = BoxPlot(y_label=get_y_label(), x_label="Set", name=name)
    plot.set_data(df_values)

    return plot


def create_violin_plot(df_values, name):
    plot = ViolinPlot(y_label=get_y_label(), x_label="Set", name=name)
    plot.set_data(df_values)

    return plot


def create_swarm_plot(df_values, name):
    plot = SwarmPlot(y_label=get_y_label(), x_label="Set", name=name)
    plot.set_data(df_values)

    return plot


def scale_df_values(df_values, scaler):
    return scaler.transform(df_values[:, np.newaxis]).flatten()


def calculate_df_values_of_rule_filters(estimator, df_values, data, scaler):
    df_values = estimator.decision_function(data.samples)
    rf_label_idcs = np.where(data.labels == 1)
    rf_df_values = df_values[rf_label_idcs]

    if scale_df and scaler:
        rf_df_values = scale_df_values(rf_df_values, scaler)

    return rf_df_values


def calculate_df_values_of_evasions(df_values, data, scaler):
    evasion_label_idcs = np.where(data.labels == 1)
    evasion_df_values = df_values[evasion_label_idcs]

    if scale_df and scaler:
        evasion_df_values = scale_df_values(evasion_df_values, scaler)

    return evasion_df_values


def calculate_df_values_of_benigns(df_values, data, scaler):
    benign_label_idcs = np.where(data.labels == 0)
    benign_df_values = df_values[benign_label_idcs]

    if scale_df and scaler:
        benign_df_values = scale_df_values(benign_df_values, scaler)

    return benign_df_values


def calculate_df_values_and_labels(result):
    df_values = {}

    if "benigns" in data:
        df_values["Benigns"] = calculate_df_values_of_benigns(
            result.predict, result.data.validation_data, result.scaler
        )
    if "evasions" in data:
        df_values["Evasions"] = calculate_df_values_of_evasions(
            result.predict, result.data.validation_data, result.scaler
        )
    if "rules" in data:
        df_values["Rules"] = calculate_df_values_of_rule_filters(
            result.estimator, result.predict, result.data.train_data, result.scaler
        )

    return df_values


def create_plot(df_values, name):
    if plot_type == "swarm":
        return create_swarm_plot(df_values, name)
    elif plot_type == "box":
        return create_box_plot(df_values, name)
    elif plot_type == "violin":
        return create_violin_plot(df_values, name)


def create_plot_from_valid_result(result):
    df_values = calculate_df_values_and_labels(result)

    return create_plot(df_values, result.name)


def create_df_values_plot():
    result = load_result(result_path)

    if type(result) is ValidationResult:
        plot = create_plot_from_valid_result(result)

    del result

    plot.plot()

    plot.save(f"{os.getcwd()}/{file_name}.pdf")


def parse_args_and_options(args):
    global result_path
    result_path = args.valid_result

    global file_name
    file_name = args.file_name

    if args.scale_df:
        global scale_df
        scale_df = True

    if args.plot_type:
        global plot_type
        plot_type = args.plot_type

    if args.data:
        global data
        data = args.data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "valid_result",
        type=str,
        action="store",
        help="Path of result file containing trained (or calibrated) estimator",
    )
    parser.add_argument(
        "file_name", type=str, action="store", help="Name of the output file"
    )
    parser.add_argument(
        "--scale-df",
        action="store_true",
        help="Scale df-values using min-max-scaler (if available)",
    )
    parser.add_argument(
        "--plot-type",
        action="store",
        choices=["swarm", "box", "violin"],
        default="box",
        help="Specify the plot type",
    )
    parser.add_argument(
        "--data",
        action="append",
        choices=["benigns", "rules", "evasions"],
        help="Specify input data for decision function values",
    )
    args = parser.parse_args()
    parse_args_and_options(args)

    create_df_values_plot()


if __name__ == "__main__":
    main()
