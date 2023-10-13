import json
import pytest
import os
import numpy as np
import ndjson

from io import TextIOWrapper
from zipfile import ZipFile, ZIP_DEFLATED
from sklearn.svm import SVC

from amides.persist import Dumper, PersistError, EventWriter, EventCompressor
from amides.data import (
    MultiTrainingResult,
    TrainTestSplit,
    DataBunch,
    TrainTestValidSplit,
    TrainingResult,
    ValidationResult,
)
from amides.models.baseline.baseline import BaselineClassifier


@pytest.fixture
def dump_dir(tmpdir):
    return tmpdir.mkdir("results")


class TestDumper:
    @pytest.fixture
    def monkeypatch_timestamp(self, monkeypatch):
        def mock_get_current_timestamp(format):
            return "19700101_000000"

        monkeypatch.setattr(
            "classifier.data.get_current_timestamp", mock_get_current_timestamp
        )

        monkeypatch.setattr(
            "classifier.models.baseline.get_current_timestamp",
            mock_get_current_timestamp,
        )

    @pytest.fixture
    def tt_split(self):
        return TrainTestSplit(
            DataBunch(np.array([[1], [0], [2], [3], [1]]), np.array([0, 0, 1, 1, 0])),
            DataBunch(np.array([[0], [2], [3], [1], [1]]), np.array([0, 1, 1, 0, 0])),
            name="some_tt_data",
        )

    @pytest.fixture
    def ttv_split(self):
        return TrainTestValidSplit(
            DataBunch(np.array([[1], [0], [2], [3], [1]]), np.array([0, 0, 1, 1, 0])),
            DataBunch(np.array([[0], [2], [3], [1], [1]]), np.array([0, 1, 1, 0, 0])),
            DataBunch(np.array([[0], [1], [0], [2], [3]]), np.array([0, 0, 0, 1, 1])),
            name="some_ttv_data",
        )

    @pytest.fixture
    def train_result(self, tt_split):
        return TrainingResult(
            SVC(), tt_split, name="some_train_result", timestamp="19700101_000000"
        )

    @pytest.fixture
    def valid_result(self, tt_split):
        return ValidationResult(
            SVC(),
            tt_split,
            np.array([1, 0, 0]),
            name="some_valid_result",
            timestamp="19700101_000000",
        )

    @pytest.fixture
    def multi_result(self, tt_split):
        multi_rslt = MultiTrainingResult(name="some_rules", timestamp="19700101_000000")
        multi_rslt.add_result(
            ValidationResult(
                SVC(),
                tt_split,
                np.array([1, 0, 0]),
                name="some_rule",
                timestamp="19700101_000000",
            )
        )
        multi_rslt.add_result(
            ValidationResult(
                SVC(),
                tt_split,
                np.array([1, 0, 0]),
                name="another_rule",
                timestamp="19700101_000000",
            )
        )

        return multi_rslt

    @pytest.fixture
    def baseline_clf(self):
        return BaselineClassifier(
            {
                "remove_escape_characters": True,
                "delete_whitespaces": True,
                "modify_exe": True,
                "swap_slash_minus": True,
                "swap_minus_slash": True,
            },
            name="some_base_clf",
            timestamp="19700101_000000",
        )

    def test_init(self, dump_dir):
        dumper = Dumper(dump_dir)
        assert dumper

    def test_init_non_existing_dir(self, tmpdir):
        output_dir = tmpdir.join("results")
        dumper = Dumper(output_dir)
        assert dumper
        assert os.path.isdir(output_dir)

    def test_save_tt_split(self, dump_dir, tt_split):
        expected_filename = "tt_split_some_tt_data"

        dumper = Dumper(dump_dir)
        dumper.save_object(tt_split)

        entries = sorted(os.listdir(dump_dir))
        assert len(entries) == 2
        assert entries[0] == f"{expected_filename}.zip"
        assert entries[1] == f"{expected_filename}_info.json"

    def test_save_ttv_split(self, dump_dir, ttv_split):
        expected_filename = "ttv_split_some_ttv_data"

        dumper = Dumper(dump_dir)
        dumper.save_object(ttv_split)

        entries = sorted(os.listdir(dump_dir))
        assert len(entries) == 2
        assert entries[0] == f"{expected_filename}.zip"
        assert entries[1] == f"{expected_filename}_info.json"

    def test_save_training_result(self, dump_dir, train_result):
        expected_filename = "train_rslt_some_train_result_19700101_000000"

        dumper = Dumper(dump_dir)
        dumper.save_object(train_result)

        entries = sorted(os.listdir(dump_dir))
        assert len(entries) == 2
        assert entries[0] == f"{expected_filename}.zip"
        assert entries[1] == f"{expected_filename}_info.json"

    def test_save_validation_result(self, dump_dir, valid_result):
        expected_filename = "valid_rslt_some_valid_result_19700101_000000"

        dumper = Dumper(dump_dir)
        dumper.save_object(valid_result)

        entries = sorted(os.listdir(dump_dir))
        assert len(entries) == 2
        assert entries[0] == f"{expected_filename}.zip"
        assert entries[1] == f"{expected_filename}_info.json"

    def test_save_calibration_result(self, dump_dir, calib_result):
        expected_filename = "calib_rslt_some_calib_result_19700101_000000"

        dumper = Dumper(dump_dir)
        dumper.save_object(calib_result)

        entries = sorted(os.listdir(dump_dir))
        assert len(entries) == 2
        assert entries[0] == f"{expected_filename}.zip"
        assert entries[1] == f"{expected_filename}_info.json"

    def test_save_multi_result(self, dump_dir, multi_result):
        expected_filename = "multi_rslt_some_rules_19700101_000000"

        dumper = Dumper(dump_dir)
        dumper.save_object(multi_result)

        entries = sorted(os.listdir(dump_dir))
        assert len(entries) == 2
        assert entries[0] == f"{expected_filename}.zip"
        assert entries[1] == f"{expected_filename}_info.json"

    def test_save_baseline_clf(self, dump_dir, baseline_clf):
        expected_filename = "baseline_clf_some_base_clf_19700101_000000"

        dumper = Dumper(dump_dir)
        dumper.save_object(baseline_clf)

        entries = sorted(os.listdir(dump_dir))
        assert len(entries) == 2
        assert entries[0] == f"{expected_filename}.zip"
        assert entries[1] == f"{expected_filename}_info.json"

    def test_save_no_output_path(self, tmpdir, tt_split):
        with tmpdir.as_cwd() as cwd:
            dumper = Dumper()
            dumper.save_object(tt_split)

        results_dir = tmpdir.join("pickled")
        entries = sorted(os.listdir(results_dir))
        assert len(entries) == 2

    def test_load_train_test_split(self, dump_dir, tt_split):
        dumper = Dumper(dump_dir)
        dumper.save_object(tt_split)

        entries = sorted(os.listdir(dump_dir))

        result = dumper.load_object(entries[0])
        assert isinstance(result, TrainTestSplit)
        assert np.array_equal(result.train_data.samples, tt_split.train_data.samples)
        assert np.array_equal(result.train_data.labels, tt_split.train_data.labels)
        assert np.array_equal(result.test_data.samples, tt_split.test_data.samples)
        assert np.array_equal(result.test_data.labels, tt_split.test_data.labels)

    def test_load_train_test_valid_split(self, dump_dir, ttv_split):
        dumper = Dumper(dump_dir)
        dumper.save_object(ttv_split)

        entries = sorted(os.listdir(dump_dir))

        result = dumper.load_object(entries[0])
        assert isinstance(result, TrainTestValidSplit)
        assert np.array_equal(result.train_data.samples, ttv_split.train_data.samples)
        assert np.array_equal(result.train_data.labels, ttv_split.train_data.labels)
        assert np.array_equal(result.test_data.samples, ttv_split.test_data.samples)
        assert np.array_equal(result.test_data.labels, ttv_split.test_data.labels)
        assert np.array_equal(
            result.validation_data.samples, ttv_split.validation_data.samples
        )
        assert np.array_equal(
            result.validation_data.labels, ttv_split.validation_data.labels
        )

    def test_load_training_result(self, dump_dir, train_result):
        dumper = Dumper(dump_dir)
        dumper.save_object(train_result)

        entries = sorted(os.listdir(dump_dir))

        result = dumper.load_object(entries[0])
        assert type(result) is TrainingResult
        assert result.estimator.get_params() == train_result.estimator.get_params()
        assert np.array_equal(
            result.data.train_data.samples, train_result.data.train_data.samples
        )
        assert np.array_equal(
            result.data.train_data.labels, train_result.data.train_data.labels
        )
        assert np.array_equal(
            result.data.test_data.samples, train_result.data.test_data.samples
        )
        assert np.array_equal(
            result.data.test_data.labels, train_result.data.test_data.labels
        )

    def test_load_validation_result(self, dump_dir, valid_result):
        dumper = Dumper(dump_dir)
        dumper.save_object(valid_result)

        entries = sorted(os.listdir(dump_dir))

        result = dumper.load_object(entries[0])
        assert type(result) is ValidationResult
        assert valid_result.estimator.get_params() == result.estimator.get_params()
        assert np.array_equal(
            result.data.train_data.samples, valid_result.data.train_data.samples
        )
        assert np.array_equal(
            result.data.train_data.labels, valid_result.data.train_data.labels
        )
        assert np.array_equal(
            result.data.test_data.samples, valid_result.data.test_data.samples
        )
        assert np.array_equal(
            result.data.test_data.labels, valid_result.data.test_data.labels
        )

    def test_load_calibration_result(self, dump_dir, calib_result):
        dumper = Dumper(dump_dir)
        dumper.save_object(calib_result)

        entries = sorted(os.listdir(dump_dir))

        result = dumper.load_object(entries[0])
        assert type(result) is CalibrationResult
        assert np.array_equal(
            result.data.train_data.samples, calib_result.data.train_data.samples
        )
        assert np.array_equal(
            result.data.train_data.labels, calib_result.data.train_data.labels
        )
        assert np.array_equal(
            result.data.test_data.samples, calib_result.data.test_data.samples
        )
        assert np.array_equal(
            result.data.test_data.labels, calib_result.data.test_data.labels
        )

    def test_load_multi_result(self, dump_dir, multi_result):
        dumper = Dumper(dump_dir)
        dumper.save_object(multi_result)

        entries = sorted(os.listdir(dump_dir))

        loaded_result = dumper.load_object(entries[0])
        assert type(loaded_result) is MultiTrainingResult

        result_1 = loaded_result.get_result("some_rule")
        result_2 = loaded_result.get_result("another_rule")

        expected_result_1 = multi_result.get_result("some_rule")
        expected_result_2 = multi_result.get_result("another_rule")

        assert np.array_equal(
            result_1.data.train_data.samples, expected_result_1.data.train_data.samples
        )
        assert np.array_equal(
            result_1.data.train_data.labels, expected_result_1.data.train_data.labels
        )
        assert np.array_equal(
            result_1.data.test_data.samples, expected_result_1.data.test_data.samples
        )
        assert np.array_equal(
            result_1.data.test_data.labels, expected_result_1.data.test_data.labels
        )

        assert np.array_equal(
            result_2.data.train_data.samples, expected_result_2.data.train_data.samples
        )
        assert np.array_equal(
            result_2.data.train_data.labels, expected_result_2.data.train_data.labels
        )
        assert np.array_equal(
            result_2.data.test_data.samples, expected_result_2.data.test_data.samples
        )
        assert np.array_equal(
            result_2.data.test_data.labels, expected_result_2.data.test_data.labels
        )

    def test_load_baseline_clf(self, dump_dir, baseline_clf):
        dumper = Dumper(dump_dir)
        dumper.save_object(baseline_clf)

        entries = sorted(os.listdir(dump_dir))

        result = dumper.load_object(entries[0])
        assert type(result) is BaselineClassifier
        assert result.modifier_mask == 0

    def test_load_object_from_dir_path(self, dump_dir):
        dumper = Dumper()

        with pytest.raises(PersistError):
            dumper.load_object(dump_dir)

    def test_load_object_absolute_path(self, dump_dir, tt_split):
        dumper = Dumper(dump_dir)
        dumper.save_object(tt_split)

        entries = os.listdir(dump_dir)
        file_path = os.path.join(dump_dir, entries[1])

        result = dumper.load_object(file_path)
        assert result

    def test_load_object_no_output_path(self, tmpdir, tt_split):
        with tmpdir.as_cwd() as cwd:
            dumper = Dumper()
            dumper.save_object(tt_split)

            entries = os.listdir(os.path.join(os.getcwd(), "pickled"))

            new_dumper = Dumper()
            result = new_dumper.load_object(entries[1])
            assert result

    def test_load_object_non_existing_file(self, dump_dir):
        dumper = Dumper()

        with pytest.raises(PersistError):
            _ = dumper.load_object(os.path.join(dump_dir, "sample.zip"))


class TestEventWriter:
    def test_init(self, tmp_path):
        test_out = tmp_path / "test"
        _ = EventWriter(
            str(test_out),
            "2023-06-01T00:00:00",
            "2023-06-01T01:00:00",
        )

        assert test_out.is_dir()

    def test_init_out_dir_exists(self, tmp_path):
        test_out = tmp_path / "test"
        test_out.mkdir()

        _ = EventWriter(
            str(test_out),
            "2023-06-01T00:00:00",
            "2023-06-01T01:00:00",
        )

        assert test_out.is_dir()

    def test_init_missing_parents_dir(self, tmp_path):
        parent_test_out = tmp_path / "test"
        test_out = parent_test_out / "sub"

        _ = EventWriter(
            str(test_out),
            "2023-06-01T00:00:00",
            "2023-06-01T01:00:00",
        )

        assert test_out.is_dir()

    def test_create_filename(self):
        result = EventWriter.create_filename(
            "2023-06-01T00:00:00", "2023-06-01T01:00:00"
        )
        expected = "events_20230601T000000_20230601T010000"

        assert result == expected

    def test_get_last_file_missing_file(self, tmp_path):
        test_out = tmp_path / "test"
        writer = EventWriter(
            str(test_out), "2023-06-01T00:00:00", "2023-06-01T01:00:00"
        )

        assert not writer.get_last_file()

    def test_get_last_file(self, tmp_path):
        test_out = tmp_path / "test"
        test_out.mkdir()
        test_file = test_out / "events_20230601T000000_20230601T010000.ndjson"
        test_file.touch()

        writer = EventWriter(
            str(test_out), "2023-06-01T00:00:00", "2023-06-01T01:00:00"
        )

        assert writer.get_last_file() == test_file

    def test_write_non_existing_file(self, tmp_path):
        test_out = tmp_path / "test"
        test_file = test_out / "events_20230601T000000_20230601T010000.ndjson"
        events = [
            {"id": "event_1", "message": "some message"},
            {"id": "event_2", "message": "another message"},
        ]
        expected = f"{ndjson.dumps(events)}\n"

        writer = EventWriter(
            str(test_out), "2023-06-01T00:00:00", "2023-06-01T01:00:00"
        )
        writer.write(events, 0)

        assert test_file.is_file()
        with test_file.open("r", encoding="utf-8") as test_file:
            result = test_file.read()

        assert result == expected

    def test_write_add_to_existing_file(self, tmp_path):
        test_out = tmp_path / "test"
        test_out.mkdir()
        test_file = test_out / "events_20230601T000000_20230601T010000.ndjson"
        events = [
            {"id": "event_1", "message": "some message"},
            {"id": "event_2", "message": "another message"},
            {"id": "evnet_3", "message": "additional message"},
        ]

        with test_file.open("w", encoding="utf-8") as out_file:
            ndjson.dump(events[:1], out_file)
            out_file.write("\n")
            out_file.flush()

        expected = f"{ndjson.dumps(events)}\n"

        writer = EventWriter(
            str(test_out), "2023-06-01T00:00:00", "2023-06-01T01:00:00"
        )
        writer.write(events[1:], 0)

        with test_file.open("r", encoding="utf-8") as in_file:
            result = in_file.read()

        assert expected == result

    def test_read_last_file_missing_file(self, tmp_path):
        test_out = tmp_path / "test"
        writer = EventWriter(
            str(test_out), "2023-06-01T00:00:00", "2023-06-01T01:00:00"
        )

        assert not writer.read_last_file()

    def test_read_last_file(self, tmp_path):
        test_out = tmp_path / "test"
        writer = EventWriter(
            str(test_out), "2023-06-01T00:00:00", "2023-06-01T01:00:00"
        )
        events = [
            {"id": "event_1", "message": "some message"},
            {"id": "event_2", "message": "another message"},
            {"id": "evnet_3", "message": "additional message"},
        ]
        expected = set([f"{json.dumps(event)}\n" for event in events])

        writer.write(events, 0)
        result = writer.read_last_file()
        assert expected == result


class TestEventCompressor:
    def test_write_non_existing_file(self, tmp_path):
        test_out = tmp_path / "test"
        test_file = test_out / "events_20230601T000000_20230601T010000.ndjson"
        test_archive = test_file.parent / f"{test_file.name}.zip"
        events = [
            {"id": "event_1", "message": "some message"},
            {"id": "event_2", "message": "another message"},
        ]
        expected = f"{ndjson.dumps(events)}\n"

        writer = EventCompressor(
            str(test_out), "2023-06-01T00:00:00", "2023-06-01T01:00:00"
        )
        writer.write(events, 0)

        assert test_archive.is_file()
        with ZipFile(str(test_archive), mode="r") as my_zip:
            with TextIOWrapper(
                my_zip.open(test_file.name, mode="r"), encoding="utf-8"
            ) as in_file:
                result = in_file.read()

        assert result == expected

    def test_write_add_to_existing_file(self, tmp_path):
        test_out = tmp_path / "test"
        test_out.mkdir()
        test_file = test_out / "events_20230601T000000_20230601T010000.ndjson"
        test_archive = test_file.parent / f"{test_file.name}.zip"
        events = [
            {"id": "event_1", "message": "some message"},
            {"id": "event_2", "message": "another message"},
            {"id": "evnet_3", "message": "additional message"},
        ]
        expected = f"{ndjson.dumps(events)}\n"

        with ZipFile(
            str(test_archive),
            mode="a",
            compression=ZIP_DEFLATED,
            allowZip64=True,
        ) as zip_file:
            with TextIOWrapper(
                zip_file.open(test_file.name, mode="w", force_zip64=True),
                encoding="utf-8",
            ) as out_file:
                ndjson.dump(events[:1], out_file)
                out_file.write("\n")
                out_file.flush()

        writer = EventCompressor(
            str(test_out), "2023-06-01T00:00:00", "2023-06-01T01:00:00"
        )
        writer.write(events[1:], 0)

        with ZipFile(str(test_archive), mode="r") as my_zip:
            with TextIOWrapper(
                my_zip.open(test_file.name, mode="r"), encoding="utf-8"
            ) as in_file:
                result = in_file.read()

        assert result == expected

    def test_read_last_file_missing_file(self, tmp_path):
        test_out = tmp_path / "test"
        writer = EventCompressor(
            str(test_out), "2023-06-01T00:00:00", "2023-06-01T01:00:00"
        )

        assert not writer.read_last_file()

    def test_read_last_file(self, tmp_path):
        test_out = tmp_path / "test"
        writer = EventCompressor(
            str(test_out), "2023-06-01T00:00:00", "2023-06-01T01:00:00"
        )
        events = [
            {"id": "event_1", "message": "some message"},
            {"id": "event_2", "message": "another message"},
            {"id": "evnet_3", "message": "additional message"},
        ]
        expected = set([f"{json.dumps(event)}\n" for event in events])

        writer.write(events, 0)
        result = writer.read_last_file()
        assert expected == result
