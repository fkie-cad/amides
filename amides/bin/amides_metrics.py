#!/usr/bin/env python3

import json
import argparse
import numpy as np


def load_amides_metrics(metrics_fp: str) -> list[dict]:
    amides_metrics = []
    pipeline_counter = 1

    for line in open(metrics_fp, "r", encoding="utf-8"):
        try:
            current_pipeline = f"pipeline-{pipeline_counter}"
            pipeline_metrics = json.loads(line)
            amides_metrics.append(
                pipeline_metrics["pipeline"][current_pipeline]["processor"]["amides"]
            )
        except KeyError:
            print(f"Pipeline {pipeline_counter} did not process any events")

        pipeline_counter += 1

    return amides_metrics


def calculate_pipeline_metrics(metrics_file):
    amides_metrics = load_amides_metrics(metrics_file)

    number_of_processed_events = []
    mean_processing_times_per_event = []
    num_cmdlines = []
    num_cached_results, num_new_results = [], []
    mean_misuse_detection_times = []
    mean_rule_attribution_times = []
    num_cache_entries, cache_load = [], []

    for metrics in amides_metrics:
        number_of_processed_events.append(
            metrics["logprep_processor_number_of_processed_events"]
        )
        mean_processing_times_per_event.append(
            metrics["logprep_processor_mean_processing_time_per_event"]
        )
        num_cmdlines.append(metrics["logprep_processor_total_cmdlines"])
        num_new_results.append(metrics["logprep_processor_new_results"])
        num_cached_results.append(metrics["logprep_processor_cached_results"])
        num_cache_entries.append(metrics["logprep_processor_num_cache_entries"])
        cache_load.append(metrics["logprep_processor_cache_load"])
        mean_misuse_detection_times.append(
            metrics["logprep_processor_mean_misuse_detection_time"]
        )
        mean_rule_attribution_times.append(
            metrics["logprep_processor_mean_rule_attribution_time"]
        )

    print(f"Number of processed events: {np.sum(number_of_processed_events):.1f}")
    print(
        f"Mean processing time per event: {np.mean(mean_processing_times_per_event):.8f} (STD: {np.std(mean_processing_times_per_event):.8f})"
    )
    print(f"Total number of command lines: {np.sum(num_cmdlines):.1f}")
    print(f"Total number of cached results: {np.sum(num_cached_results)}")
    print(f"Total number  new results: {np.sum(num_new_results):.1f}")
    print(f"Total number of cache entries: {np.sum(num_cache_entries):.2f}")
    print(f"Average cache load: {np.average(cache_load):.4f}")
    print(
        f"Mean misuse detection time: {np.mean(mean_misuse_detection_times):.8f} (STD: {np.std(mean_misuse_detection_times):.8f})"
    )
    print(
        f"Mean rule attribution:time: {np.mean(mean_rule_attribution_times):.8f} (STD: {np.std(mean_rule_attribution_times):.8f})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metrics_file",
        action="store",
        help="Path to the performance metrics file (.jsonl)",
    )
    args = parser.parse_args()

    calculate_pipeline_metrics(args.metrics_file)


if __name__ == "__main__":
    main()
