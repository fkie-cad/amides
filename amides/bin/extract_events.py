#!/usr/bin/env python3

import yaml
import argparse
import json

from pathlib import Path
from logging import Logger

from generic_log_extractor.run_generic_log_extractor import GenericLogExtractor
from generic_log_extractor.utils.logger import create_logger
from generic_log_extractor.connectors.connector_factory import ConnectorFactory
from generic_log_extractor.connectors.elasticsearch_connector import Elasticsearch
from generic_log_extractor.connectors.opensearch_connector import OpenSearch
from generic_log_extractor.run_generic_log_extractor import (
    _generate_sub_dict,
    _check_key_in_dict,
)

from amides.utils import TimeRangeIterator
from amides.persist import EventCompressor


class CacheExtractor(GenericLogExtractor):
    """SimpleExtractor to override '_ensure_file_has_unique_records'."""

    def __init__(
        self,
        indices: list[str],
        lucene_filter: str,
        timerange: dict,
        keep: list[str],
        connector: tuple[OpenSearch, Elasticsearch],
        file_writer: EventCompressor,
        logger: Logger,
    ):
        super().__init__(
            indices=indices,
            lucene_filter=lucene_filter,
            timerange=timerange,
            keep=keep,
            connector=connector,
            file_writer=file_writer,
            logger=logger,
        )
        self._records = []

    def extract(self) -> int:
        """
        Retrieves records from elasticsearch/opensearch and writes results into files.

        Parameters
        ----------
        file_writer: FileWriter
            Writes records into files.

        Returns
        -------
        int
            Count of results.

        """

        index_names = self._get_index_names()
        completed_indices = []
        results_per_index = {index: [] for index in index_names}

        self._last_timestamp_per_index = {
            index: self._base_start_time for index in index_names
        }
        self._last_ids_per_index = {index: [] for index in index_names}

        batch_size = 10000
        for index_name in index_names:
            success = False
            while not success:
                try:
                    self._process_index(batch_size, index_name, results_per_index)
                    completed_indices.append(index_name)
                    self._reset_query_timerange()
                    self._resuming = False
                    success = True
                except BaseException as error:  # pylint: disable=broad-except
                    self._logger.exception(error)
                    self._logger.info(
                        f"Resuming extraction at timestamp "
                        f"'{self._last_timestamp_per_index[index_name]}'"
                    )
                    self._update_query_timerange(index_name)
                    self._connector.clear_scroll(scroll_id=self._scroll_id)
                    self._resuming = True

        self._file_writer.write(self._records, 0)
        results_total = self._get_total_results(results_per_index)

        return len(results_total)

    def _process_results(
        self, batch_size: int, index_name: str, res: dict, results_per_index: dict
    ) -> int:
        self._scroll_id = res["_scroll_id"]
        hits = [hit["_source"] for hit in res["hits"]["hits"]]
        scroll_size = len(hits)

        self._log_retrieved(scroll_size)

        if self._resuming and hits:
            old_hit_str = {json.dumps(hit) for hit in self._records}
            hit_str = {json.dumps(hit) for hit in hits}
            hit_str_diff = [hit for hit in hit_str if hit not in old_hit_str]
            hits = list(json.loads(hit) for hit in hit_str_diff)
            scroll_size = len(hits)

        if hits:
            if self.keep:
                sparse_hits = []
                for hit in hits:
                    existing_keys = [
                        key for key in self.keep if _check_key_in_dict(hit, key)
                    ]
                    sparse_hits.append(_generate_sub_dict(hit, existing_keys))
                self._records.extend(sparse_hits)
            else:
                self._records.extend(hits)
            try:
                self._last_timestamp_per_index[index_name] = hits[-1]["@timestamp"]
            except KeyError:
                self._last_timestamp_per_index[index_name] = hits[-1]["timestamp"]
            results_per_index[index_name] += hits

        if self._resuming:
            self._ensure_records_are_unique()

        self._resuming = False
        return scroll_size

    def _ensure_records_are_unique(self):
        unique_records = set(json.dumps(record) for record in self._records)
        self._records = list(json.loads(record) for record in unique_records)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Configuration file (default: config.yml)",
        default=Path("config.yml"),
        type=Path,
    )

    return parser.parse_args()


def extract_events(config: dict):
    logger = create_logger(config.get("logging_level", "INFO"))
    out_dir = Path(config["output_directory"])

    extraction = config["extraction"][0]
    connector = ConnectorFactory.get_connectors(config, logger)[extraction["connector"]]
    indices = extraction["indices"]
    lucene_filter = extraction.get("lucene_filter", "")
    timerange = extraction.get("iso8601_timerange")
    keep_fields = extraction.get("fields")

    time_ranger = TimeRangeIterator(
        timerange["start"], timerange["end"], timerange["interval"]
    )
    for start, end in time_ranger.next():
        file_writer = EventCompressor(out_dir=out_dir, start=start, end=end)
        extractor = CacheExtractor(
            indices=indices,
            lucene_filter=lucene_filter,
            timerange={"start": start, "end": end},
            keep=keep_fields,
            connector=connector,
            file_writer=file_writer,
            logger=logger,
        )

        events_count = extractor.extract()
        logger.info(
            "Retrieved %s records in total from indices: %s for interval %s - %s",
            events_count,
            ",".join(indices),
            start,
            end,
        )


def main():
    args = parse_arguments()
    with args.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    extract_events(config)


if __name__ == "__main__":
    main()
