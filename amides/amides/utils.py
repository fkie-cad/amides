""" This module contains functions for general purpose use, e.g. opening files or fetching directory contents. 
"""
import os
import re
import json
import logging
import functools
import time
import yaml
import argparse

from datetime import datetime, timedelta
from dateutil import parser


_log_level = logging.INFO


class TimeRangeIterator:
    """TimeRangeIterator returns time range intervals from 'start' to
    'end' in specified intervals.
    """

    def __init__(self, start: str, end: str, interval: str):
        """Create TimeRangeIterator.

        Parameters
        ----------
        start : str
            Starting timestamp in ISO8601 format
        end: str
            Ending timestamp in ISO8601 format
        interval: str
            Interval in 'HH:MM:SS.s+'
        """
        self._start = self._parse_timestamp(start)
        self._end = self._parse_timestamp(end)

        self._interval = self._parse_interval(interval)

    def next(self):
        """Returns the next timestamp value."""
        current_start = self._start
        current_end = current_start + self._interval
        yield current_start.isoformat(), current_end.isoformat()

        while current_end < self._end:
            current_start = current_end
            current_end += self._interval
            yield current_start.isoformat(), current_end.isoformat()

    def _parse_timestamp(self, timestamp: str):
        return parser.parse(timestamp)

    def _parse_interval(self, interval: str):
        m = re.match(
            r"(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d[\.\d+]*)", interval
        )
        converted = {key: float(value) for key, value in m.groupdict().items()}

        return timedelta(**converted)


def load_args_from_file(
    parser: argparse.ArgumentParser, path: str
) -> argparse.Namespace:
    """Loads command line arguments from config file and
    puts values in args.Namsespace object.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ArgumentParser instance
    path: str
        Path of the configuration file (.json)

    Returns
    -------
    args: Optional[argparse.ArgumentParser]
    """
    config_dict = read_json_file(path)
    if config_dict:
        ns = argparse.Namespace()
        ns.__dict__.update(config_dict)
        args = parser.parse_args(namespace=ns)

        return args
    return None


def get_current_timestamp(ts_format="%Y%m%d_%H%M%S"):
    """
    Returns current timestamp according to the specified format.

    Parameters
    ----------
    format: str
        String representing the required timestamp format.

    Returns
    -------
    timestamp: str
        String representation of the current timestamp.

    """
    return datetime.now().strftime(ts_format)


def get_file_names(path):
    """
    Return list of file names of the specified directory path.

    Parameters
    ----------
    path: str
        Path of the target directory. _logger.error("Error reading config file. Exiting")
        sys.exit(1)

    Returns
    -------
    file_names: List[str]
        List of file names.
    """
    file_names = []
    for entry in sorted(os.listdir(path)):
        if os.path.isfile(os.path.join(path, entry)):
            file_names.append(entry)

    return file_names


def get_file_paths(path):
    """
    Returns list of file paths of files located in specified directory.

    Parameters
    ----------
    path: str
        Path of the target directory.

    Returns
    -------
    paths: List[str]
        List of file paths.
    """
    file_paths = []
    for entry in sorted(os.listdir(path)):
        entry_path = os.path.join(path, entry)
        if os.path.isfile(entry_path):
            file_paths.append(entry_path)

    return file_paths


def get_dir_names(path):
    """
    Return list of directories under specified target path.

    Parameters
    ----------
    path: str
        Path of the target directory.

    Returns
    -------
    dir_names: List[str]
        List of directory names.
    """
    dir_names = []
    for entry in sorted(os.listdir(path)):
        entry_path = os.path.join(path, entry)
        if os.path.isdir(entry_path):
            dir_names.append(entry)

    return dir_names


def read_yaml_file(path):
    """
    Expects and reads contents of .y(a)ml-file at specified target path.

    Parameters
    ----------
    path: str
        Path to the .y(a)ml-file.

    Returns
    -------
    yaml-data: Dict
        Dict-representation of the loaded .y(a)ml-file.

    """

    with open(path, "r", encoding="utf-8") as f:
        docs = yaml.safe_load_all(f)
        return list(docs)


def read_json_file(path):
    """
    Expects and reads .json-file at specified target path.

    Parameters
    ----------
    path: str
        Path of the .json-file.

    Returns
    -------
    json-data: Optional[Dict]
        Dictionary containing JSON-data or None.

    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

            return data
    except (FileNotFoundError, json.JSONDecodeError) as err:
        _logger.error(err)
        return None


def read_jsonl_file(path):
    """
    Reads json objects from .jsonl-file.

    Parameters
    ----------
    path: str
        Path of the expected .jsonl-file.

    Returns
    -------
    json_objects: Optional[List[Dict]]
        List of Dictionaries holding JSON-data or None.

    """
    try:
        json_objects = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                json_objects.append(json.loads(line))

        return json_objects
    except (FileNotFoundError, json.JSONDecodeError) as err:
        _logger.error(err)
        return None


def execution_time(func):
    """
    Wrapper for functions and methods that measures execution time in
    milliseconds (ms)

    Parameters
    ----------
    func: callable
        Function whose execution time should be measured

    Returns
    -------
    result: object
        Return values of the executed function
    """
    wrapped_function = func
    while hasattr(wrapped_function, "__wrapped__"):
        wrapped_function = wrapped_function.__wrapped__

    target_logger = wrapped_function.__globals__.get("_logger")
    if target_logger is None:
        target_logger = logging.getLogger("")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        target_logger.info("Entering %s", func.__name__)
        enter_time = time.time()

        result = func(*args, **kwargs)

        return_time = time.time()

        duration = (return_time - enter_time) * 1000.0
        target_logger.info(
            "Returning after {0:.2f}ms from {1}".format(duration, func.__name__)
        )

        return result

    return wrapper


def set_log_level(log_level):
    """Sets the global log level of all loggers.

    Parameters
    ----------
    log_level: str
        String representation of the required log level.

    Raises
    ------
    ValueError
        In case the specified log level string does not correspond
        to any known log level.
    """
    global _log_level
    log_level = log_level.lower()

    if log_level == "info":
        _log_level = logging.INFO
    elif log_level == "debug":
        _log_level = logging.DEBUG
    elif log_level == "warning":
        _log_level = logging.WARNING
    elif log_level == "error":
        _log_level = logging.ERROR
    elif log_level == "critical":
        _log_level = logging.CRITICAL
    else:
        raise ValueError(f"Log level '{log_level}' is not a known log level")


def get_logger(name):
    """Creates and returns logger with the specified log level

    Parameters
    ----------
    name: str
        Name of the logger instance

    Returns
    -------
    logger: logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level=_log_level)

    console_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(process)d-%(thread)x] %(name)s: %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=_log_level)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(console_handler)

    return logger


_logger = get_logger(__name__)
