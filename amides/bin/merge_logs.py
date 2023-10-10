#!/usr/bin/env python3

import ndjson
import argparse
import pandas as pds
import re
import json
import matplotlib.pyplot as plt

from pathlib import Path


def get_max_y_value(metrics: dict[str, pds.DataFrame]):
    tmp_concat = pds.concat(list(metrics.values()))
    max_row = tmp_concat.max(axis=0)

    return max_row.max()


def plot_performance_metrics_per_pipeline(pipeline_metrics: dict[str, pds.DataFrame]):
    fig, axs = plt.subplots(
        nrows=len(pipeline_metrics),
        figsize=(6, 13),
        sharex=True,
        gridspec_kw={"hspace": 0.4},
    )
    y_lim_max = get_max_y_value(pipeline_metrics)

    for i, (pipeline, metrics) in enumerate(pipeline_metrics.items()):
        metrics.rename(
            columns={
                "processor.amides.logprep_processor_cached_results": "Cache Results (Hits)",
                "processor.amides.logprep_processor_new_results": "New Results (Misses)",
            },
            inplace=True,
        )
        metrics.plot(ax=axs[i], title=pipeline, legend=False)
        axs[i].set_ylim([0, y_lim_max])

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower center", ncol=len(labels))

    plt.show()


def plot_cumulated_performance_metrics(metrics: pds.DataFrame):
    fig, ax = plt.subplots(nrows=1)
    metrics.rename(
        columns={
            "processor.amides.logprep_processor_cached_results": "Cache Results (Hits)",
            "processor.amides.logprep_processor_new_results": "New Results (Misses)",
        },
        inplace=True,
    )
    metrics.plot(ax=ax, title="Cumulated", legend=True)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")

    plt.show()


def load_logs_from_ndjson_file(logs_file_path: Path):
    with open(logs_file_path, "r", encoding="utf-8") as logs_file:
        return ndjson.load(logs_file)


def load_logs_from_dir(logs_dir: str):
    logs_path = Path(logs_dir)
    logs = []

    for log_file in logs_path.iterdir():
        if log_file.is_file():
            logs.extend(load_logs_from_ndjson_file(log_file))

    return logs


def load_logs_from_out_file(logs_path: str):
    logs = []
    logprep_logger_info_regex = r"(?:\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+Logprep-JSON-File-Logger\s+INFO\s+:)(.*)"

    for line in open(logs_path, "r", encoding="utf-8"):
        match = re.match(logprep_logger_info_regex, line)
        if match:
            loaded = json.loads(match.group(1))
            logs.append(loaded)

    return logs


def load_logs(logs_path: str, from_std_out: bool) -> list[dict]:
    if from_std_out:
        return load_logs_from_out_file(logs_path)
    else:
        return load_logs_from_dir(logs_path)


def rebuild_pipeline_log_streams(logs: list[dict]) -> dict[str, list]:
    unpacked_logs = []
    for log in logs:
        unpacked_logs.append({**log["pipeline"], "timestamp": log["meta"]["timestamp"]})

    pipeline_logs = {}
    for log in unpacked_logs:
        pipeline = next(iter(log.keys()))
        logs = pipeline_logs.get(pipeline, [])
        logs.append({**log[pipeline], "timestamp": log["timestamp"]})
        pipeline_logs[pipeline] = logs

    for pipeline in pipeline_logs.keys():
        pipeline_logs[pipeline].sort(key=lambda l: l["timestamp"])

    return pipeline_logs


def get_nested_key_value(data_dict: dict, key: str) -> object:
    keys = key.split(".")
    for key in keys:
        if isinstance(data_dict, dict) and key in data_dict:
            data_dict = data_dict[key]
        else:
            return None

    return data_dict


def extract_metric_values(log: dict, metric_keys: list[str], metric_values: dict):
    for metric_key in metric_keys:
        metric_values[metric_key].append(get_nested_key_value(log, metric_key))


def extract_timestamp(log: dict, timestamps: list[str]):
    timestamps.append(log["timestamp"])


def extract_pipeline_metrics(logs: list[dict], metrics: list[str]) -> dict:
    metric_values = {metric: [] for metric in metrics}
    timestamps = []

    for log in logs:
        extract_metric_values(log, metrics, metric_values)
        extract_timestamp(log, timestamps)

    return metric_values, timestamps


def create_dataframe(metrics: dict[list], timestamps: list[str]) -> pds.DataFrame:
    df = pds.DataFrame(metrics, index=timestamps)
    df.index = pds.to_datetime(df.index)
    df.fillna(0.0, inplace=True)

    return df


def create_metrics_time_series(
    pipeline_logs: dict[list], metric_keys: list[str]
) -> list:
    pipeline_metric_time_series = {}

    for pipeline, log_series in pipeline_logs.items():
        metrics, timestamps = extract_pipeline_metrics(log_series, metric_keys)
        df = create_dataframe(metrics, timestamps)
        pipeline_metric_time_series[pipeline] = df

    return pipeline_metric_time_series


def create_cache_status_time_series(pipeline_log_streams: dict[list]) -> pds.DataFrame:
    cache_status_dfs = create_metrics_time_series(
        pipeline_log_streams,
        [
            "processor.amides.logprep_processor_cached_results",
            "processor.amides.logprep_processor_new_results",
        ],
    )

    return cache_status_dfs


def create_cumulated_cache_status_time_series(
    pipeline_log_streams: dict[list],
) -> pds.DataFrame:
    cache_status_ts = create_metrics_time_series(
        pipeline_log_streams,
        [
            "processor.amides.logprep_processor_cached_results",
            "processor.amides.logprep_processor_new_results",
        ],
    )

    dfs = list(cache_status_ts.values())
    merged_cache_status = pds.concat(dfs)
    # merged_cache_status = merged_cache_status.loc[
    #    :, ~merged_cache_status.columns.duplicated()
    # ]

    merged_cache_status.sort_index(inplace=True)
    merged_cache_status.index = pds.to_datetime(merged_cache_status.index)

    resampled = merged_cache_status.resample("5min")
    summed = resampled.sum()

    return summed


def create_performance_metrics_plots(logs_path: str, from_std_out: bool):
    logs = load_logs(logs_path, from_std_out)
    pipeline_log_streams = rebuild_pipeline_log_streams(logs)

    cache_status = create_cache_status_time_series(pipeline_log_streams)
    cumulated_cache_status = create_cumulated_cache_status_time_series(
        pipeline_log_streams
    )

    # plot_performance_metrics_per_pipeline(cache_status)
    plot_cumulated_performance_metrics(cumulated_cache_status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logs_path",
        action="store",
        help="Path to the directory containing logs",
    )
    parser.add_argument(
        "--from-std-out",
        action="store_true",
        help="Specify if logs are loaded from std-out",
    )
    args = parser.parse_args()

    create_performance_metrics_plots(args.logs_path, args.from_std_out)


if __name__ == "__main__":
    main()
