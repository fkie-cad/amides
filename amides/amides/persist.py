"""This module contains the functionality to save and load training and
validation results.
"""
import os
import json
import ndjson

from zipfile import ZipFile, ZIP_DEFLATED
from joblib import dump, load
from pathlib import Path
from typing import Optional
from io import TextIOWrapper

from amides.data import DataSplit, TrainingResult, MultiTrainingResult
from amides.evaluation import (
    BinaryEvaluationResult,
    RuleAttributionEvaluationResult,
)
from amides.utils import get_logger


_logger = get_logger(__name__)


class PersistError(Exception):
    """Basic PersistError exception class."""


class Dumper:
    """Dumper class to pickle and load objects holding data and training/validation results."""

    def __init__(self, output_path=None):
        """Creates object dumper.

        Parameters
        ----------
        output_path: str
            Path of the output directory used for object pickling.

        """
        self._compression_level = 9
        self._compression = ZIP_DEFLATED
        self._archive_out_format = ".zip"
        self._dictionary_out_format = ".json"
        self._output_path = None

        if output_path is not None:
            self._create_output_directory(output_path)

    def _create_output_directory(self, output_path):
        if not os.path.isdir(output_path):
            _logger.info("Creating output directory at %s", output_path)
            os.makedirs(output_path, exist_ok=True)
            _logger.info("Created output directory at %s", output_path)

        self._output_path = output_path

    def save_object(self, obj, file_name=None):
        """Save given object.

        Parameters
        ----------
        obj: object
            Object which should be pickled.

        file_name: Optional[str]
            Name of the output file.

        """
        if self._is_known_object(obj):
            self._save_known_object(obj, file_name)
        else:
            self._save_unknown_object(obj, file_name)

    def _save_known_object(self, obj, file_name):
        if file_name is None:
            file_name = obj.file_name()

        _logger.info("Saving %s %s", type(obj), file_name)
        self._dump_object(obj, file_name)
        _logger.info("Saved %s %s", type(obj), file_name)

        if callable(getattr(obj, "create_info_dict", None)):
            info_dict = obj.create_info_dict()
            info_file_name = f"{file_name}_info"

            _logger.info("Saving info-dict %s", info_file_name)
            self._dump_dictionary(info_dict, info_file_name)
            _logger.info("Saved info-dict %s", info_file_name)

    def _save_unknown_object(self, obj, file_name):
        if file_name is None:
            if not callable(getattr(obj, "file_name", None)):
                raise PersistError(
                    "No file name and method to create file name available"
                )

            file_name = obj.file_name()
            if not isinstance(file_name, str):
                raise PersistError(
                    "Function file_name() of unknown object does not return valid file name string"
                )

        self._dump_object(obj, file_name)

    def create_out_file_path(self, file_name):
        """Builds path for an external output file using the dumpers
        output directory.

        Parameters
        ----------
        file_name: str
            Name of the external file.

        Returns
        -------
        path: str
            Complete path to the file in the Dumper's output directory.

        """
        if self._output_path is None:
            output_directory_path = os.path.join(os.getcwd(), "pickled")
            _logger.warning(
                "No output path specified. Creating output directory %s",
                output_directory_path,
            )
            self._create_output_directory(output_directory_path)

        return os.path.join(self._output_path, file_name)

    def load_object(self, object_path):
        """Loads pickled object from specified path

        Parameters
        ----------
        object_path: str
            Path of the pickled object.

        Raises
        ------
        TypeError
            If the loaded object is not of the expected type.

        """
        _logger.info("Loading object from %s", object_path)
        loaded_object = self._load_object(object_path)
        _logger.info("Loaded object from %s", object_path)

        if not self._is_known_object(loaded_object):
            raise PersistError("Loaded object is not of any knwon object types")

        return loaded_object

    def _dump_object(self, obj, file_name):
        path = self.create_out_file_path(f"{file_name}{self._archive_out_format}")
        _logger.debug("Dumping object %s at %s", file_name, path)

        with ZipFile(
            path, mode="w", compression=self._compression, allowZip64=True
        ) as zip_file:
            with zip_file.open(file_name, mode="w", force_zip64=True) as out_file:
                dump(obj, out_file, protocol=4)

        _logger.debug("Dumped object %s at %s", file_name, path)

    def _dump_dictionary(self, dictionary, file_name):
        path = self.create_out_file_path(f"{file_name}{self._dictionary_out_format}")
        _logger.debug("Dumping dictionary %s at %s", file_name, path)

        with open(path, "w", encoding="utf-8") as out_file:
            json.dump(dictionary, out_file, ensure_ascii=False, indent=4)

        _logger.debug("Dumped dictionary %s at %s", file_name, path)

    def _is_known_object(self, obj):
        if not isinstance(
            obj,
            (
                DataSplit,
                TrainingResult,
                MultiTrainingResult,
                BinaryEvaluationResult,
                RuleAttributionEvaluationResult,
                dict,
            ),
        ):
            return False
        return True

    def _load_object(self, object_path):
        if os.path.isdir(object_path):
            raise PersistError(f"Specified object path {object_path} is a directory")

        if not (os.path.isfile(object_path) or os.path.isabs(object_path)):
            object_path = self.create_out_file_path(object_path)

        file_name = os.path.basename(object_path)
        file_name_without_ending = file_name.rstrip(".zip")

        try:
            with ZipFile(object_path, mode="r") as my_zip:
                with my_zip.open(file_name_without_ending, mode="r") as in_file:
                    obj = load(in_file)

            return obj
        except (FileNotFoundError, PermissionError) as err:
            raise PersistError(repr(err)) from err


class EventWriter:
    """EventWriter to write events of certain interval into single file.
    Supports methods and attributes used by the FileWriter class normally
    used by the GenericLogExtractor.
    """

    def __init__(self, out_dir: str, start: str, end: str):
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)

        self._output_path = self._out_dir / f"{self.create_filename(start, end)}.ndjson"

    @staticmethod
    def create_filename(start_iso: str, end_iso: str):
        """Create file name for the events that have been written to file."""
        start = start_iso.replace(":", "").replace("-", "")
        end = end_iso.replace(":", "").replace("-", "")

        return f"events_{start}_{end}"

    def write(self, hits: list[dict]):
        """Write events to file in batches. If file exists, events are appended.

        Parameters
        ----------
        hist: List[Dict]
            List of events that should be written to file.
        """
        with self._output_path.open("a+", encoding="utf-8") as out_file:
            self._write_batch(hits, out_file)

    def get_last_file(self) -> Optional[Path]:
        """Returns the path of the last output file.

        Returns
        -------
        :Optional[Path]
        """
        return self._output_path if self._output_path.is_file() else None

    def read_last_file(self) -> set[str]:
        """Read events already written to the last file.

        Returns
        -------
        :set[str]
            Set of unique events written to the last file.
        """
        last_file = self.get_last_file()
        if last_file:
            with last_file.open("r", encoding="utf-8") as last_file:
                return set(last_file.readlines())

        return set()

    def _write_batch(self, batch: list[dict], file_handle: TextIOWrapper):
        ndjson.dump(batch, file_handle)
        file_handle.write("\n")
        file_handle.flush()


class EventCompressor(EventWriter):
    """EventCompressor to write events of certain interval into single file.
    Supports methods and attributes used by the FileWriter class normally
    used by the GenericLogExtractor."""

    def __init__(
        self, out_dir: str, start: str, end: str, compression: str = ZIP_DEFLATED
    ):
        super().__init__(out_dir, start, end)
        self._compression = compression
        self._archive_path = self._output_path.parent / f"{self._output_path.name}.zip"

    def write(self, hits: list[dict]):
        """Write list of events to .zip-file.

        Parameters
        ----------
        hits: list[dict]
            List of dictionaries.
        """
        if self._archive_path.is_file():
            events = self._get_events()
            events.extend(hits)
            self._archive_path.unlink()
            self._write_events(events)

        else:
            self._write_events(hits)

    def get_last_file(self) -> Path:
        return self._archive_path if self._archive_path.is_file() else None

    def read_last_file(self) -> set[str]:
        last_file = self.get_last_file()
        if last_file:
            with ZipFile(self._archive_path, mode="r") as my_zip:
                with TextIOWrapper(
                    my_zip.open(self._output_path.name, mode="r"), encoding="utf-8"
                ) as in_file:
                    return set(in_file.readlines())
        return set()

    def _get_events(self) -> list[dict]:
        with ZipFile(
            self._archive_path,
            mode="a",
            compression=self._compression,
            allowZip64=True,
        ) as zip_file:
            with TextIOWrapper(
                zip_file.open(self._output_path.name, mode="r", force_zip64=True),
                encoding="utf-8",
            ) as out_file:
                return ndjson.load(out_file)

    def _write_events(self, events: list[dict]):
        with ZipFile(
            self._archive_path,
            mode="a",
            compression=self._compression,
            allowZip64=True,
        ) as zip_file:
            with TextIOWrapper(
                zip_file.open(self._output_path.name, mode="w", force_zip64=True),
                encoding="utf-8",
            ) as out_file:
                self._write_batch(events, out_file)


def get_dumper(out_dir: str) -> Dumper:
    """Returns Dumper object using specified output-path as
    output directory.

    Parameters
    ----------
    out_dir: str
        Path of the output directory.

    Returns
    -------
        : Dumper
        Dumper object.
    """
    try:
        return Dumper(out_dir)
    except OSError as err:
        raise PersistError from err
