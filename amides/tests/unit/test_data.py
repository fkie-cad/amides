import pytest
import os
import numpy as np

from amides.data import (
    DataBunch,
    TrainTestValidSplit,
    TrainTestSplit,
)


@pytest.fixture
def data_path():
    return os.path.realpath(os.path.join(os.path.dirname(__file__), "../data"))


@pytest.fixture
def benign_events_path(data_path):
    return os.path.join(data_path, "socbed-sample")


@pytest.fixture
def sigma_path(data_path):
    return os.path.join(data_path, "sigma-study")


class TestDataBunch:
    @pytest.mark.parametrize(
        "samples, labels",
        [
            (np.array(["some", "event", "data"]), np.array([0, 1, 0])),
            (np.zeros(shape=(3, 1)), np.array([0, 1, 0])),
        ],
    )
    def test_init(self, samples, labels):
        label_names = ["benign", "matching"]

        data_bunch = DataBunch(samples, labels, label_names)
        assert data_bunch

    def test_init_invalid_data_type(self):
        samples = ["some", "event", "data"]
        labels = np.array([0, 1, 0])

        with pytest.raises(TypeError):
            _ = DataBunch(samples, labels)

    def test_init_invalid_labels_type(self):
        samples = np.array(["some", "event", "data"])
        labels = [0, 1, 0]

        with pytest.raises(TypeError):
            _ = DataBunch(samples, labels)

    @pytest.mark.parametrize(
        "samples, labels",
        [
            (np.array(["some", "event", "data"]), np.array([0, 1])),
            (np.array(["some", "event"]), np.array([0, 1, 0])),
            (np.zeros(shape=(3, 1)), np.array([0, 1])),
        ],
    )
    def test_init_data_labels_shape_mismatch(self, samples, labels):
        with pytest.raises(ValueError):
            _ = DataBunch(samples, labels)

    def test_set_data_invalid_type(self):
        bunch = DataBunch(np.array(["some", "data", "points"]), np.array([0, 1, 0]))
        new_samples = [0, 1, 0]

        with pytest.raises(TypeError):
            bunch.samples = new_samples

    def test_set_labels_invalid_type(self):
        bunch = DataBunch(np.array(["some", "data", "points"]), np.array([0, 1, 0]))
        new_labels = [0, 1, 0]

        with pytest.raises(TypeError):
            bunch.labels = new_labels

    def test_set_labels_shape_mismatch(self):
        bunch = DataBunch(np.array(["some", "data", "points"]), np.array([0, 1, 0]))
        new_labels = np.array([0, 1])

        with pytest.raises(ValueError):
            bunch.labels = new_labels

    def test_stack_horizontally(self):
        bunch = DataBunch(np.array(["some", "event", "data"]), np.array([0, 1, 0]))
        other_bunch = DataBunch(
            np.array(["more", "event", "data"]), np.array([0, 1, 0])
        )

        bunch.stack_horizontally(other_bunch)
        assert bunch.samples.shape == (3, 2)
        assert np.array_equal(
            bunch.samples,
            np.array([["some", "more"], ["event", "event"], ["data", "data"]]),
        )

    def test_stack_horizontally_multiple_iterations(self):
        bunch = DataBunch(np.array(["some", "event", "data"]), np.array([0, 1, 0]))
        other_bunch = DataBunch(
            np.array(["more", "event", "data"]), np.array([0, 1, 0])
        )
        another_bunch = DataBunch(
            np.array(["even", "more", "data"]), np.array([0, 1, 0])
        )

        bunch.stack_horizontally(other_bunch)
        bunch.stack_horizontally(another_bunch)

        assert bunch.samples.shape == (3, 3)
        assert np.array_equal(
            bunch.samples,
            np.array(
                [
                    ["some", "more", "even"],
                    ["event", "event", "more"],
                    ["data", "data", "data"],
                ]
            ),
        )

    def test_stack_horizontally_mismatching_dimensions(self):
        bunch = DataBunch(np.array(["some", "event", "data"]), np.array([0, 1, 0]))
        other_bunch = DataBunch(np.array(["more", "data"]), np.array([0, 1]))

        with pytest.raises(ValueError):
            bunch.stack_horizontally(other_bunch)

    def test_stack_horizontally_invalid_input_data(self):
        bunch = DataBunch(np.array(["some", "event", "data"]), np.array([0, 1, 0]))
        other_bunch = "some-data"

        with pytest.raises(TypeError):
            bunch.stack_horizontally(other_bunch)

    def test_split(self):
        bunch = DataBunch(
            np.array(["some", "more", "event", "data"]), np.array([0, 1, 1, 0])
        )
        split_bunches = bunch.split(2, seed=42)

        assert np.array_equal(split_bunches[0].samples, np.array(["event", "some"]))
        assert np.array_equal(split_bunches[0].labels, np.array([1, 0]))
        assert np.array_equal(split_bunches[1].samples, np.array(["more", "data"]))
        assert np.array_equal(split_bunches[1].labels, np.array([1, 0]))

    def test_split_num_splits_too_low(self):
        bunch = DataBunch(
            np.array(["some", "more", "event", "data"]), np.array([0, 1, 1, 0])
        )

        with pytest.raises(ValueError):
            _, _ = bunch.split(1, seed=42)

    def test_split_num_pos_samples_too_low(self):
        bunch = DataBunch(
            np.array(["some", "more", "event", "data"]), np.array([0, 1, 0, 0])
        )

        with pytest.raises(ValueError):
            _, _ = bunch.split(2)

    def test_split_num_neg_samples_too_low(self):
        bunch = DataBunch(
            np.array(["some", "more", "event", "data"]), np.array([1, 0, 1, 1])
        )

        with pytest.raises(ValueError):
            _, _ = bunch.split(2)

    def test_split_imbalanced_data(self):
        bunch = DataBunch(
            np.array(["some", "more", "event", "data", "what"]),
            np.array([0, 1, 1, 0, 1]),
        )

        split_bunches = bunch.split(2, seed=42)
        assert np.array_equal(
            split_bunches[0].samples, np.array(["more", "event", "data"])
        )
        assert np.array_equal(split_bunches[0].labels, np.array([1, 1, 0]))
        assert np.array_equal(split_bunches[1].samples, np.array(["what", "some"]))
        assert np.array_equal(split_bunches[1].labels, np.array([1, 0]))

    def test_create_info_dict(self):
        bunch = DataBunch(
            np.array(["some", "more", "event", "data"]), np.array([1, 0, 1, 1])
        )
        bunch.add_feature_info("text")

        info_dict = bunch.create_info_dict()
        assert info_dict["shape"] == (4,)
        assert info_dict["feature_info"] == ["text"]
        assert info_dict["class_info"]["num_positive_samples"] == 3
        assert info_dict["class_info"]["num_negative_samples"] == 1
        assert info_dict["class_info"]["positive_negative_ratio"] == 3

    def test_from_binary_classification_data(self):
        elements_class_a = ["some", "class", "elements"]
        elements_class_b = ["other", "class", "elements"]
        class_names = ["class_a", "class_b"]

        bunch = DataBunch.from_binary_classification_data(
            elements_class_a, elements_class_b, class_names
        )

        assert np.array_equal(bunch.samples[:3], elements_class_a)
        assert np.array_equal(bunch.samples[3:], elements_class_b)
        assert np.array_equal(bunch.labels[:3], [0, 0, 0])
        assert np.array_equal(bunch.labels[3:], [1, 1, 1])

    def test_from_binary_classification_data_custom_class_labels(self):
        elements_class_a = ["some", "class", "elements"]
        elements_class_b = ["other", "class", "elements"]
        class_names = ["class_a", "class_b"]
        class_labels = (1, -1)

        bunch = DataBunch.from_binary_classification_data(
            elements_class_a, elements_class_b, class_names, class_labels
        )

        assert np.array_equal(bunch.labels[:3], [1, 1, 1])
        assert np.array_equal(bunch.labels[3:], [-1, -1, -1])

    def test_from_binary_classification_data_invalid_element_class(self):
        elements_class_b = ["other", "class", "elements"]
        class_names = ["class_a", "class_b"]

        with pytest.raises(TypeError):
            _ = DataBunch.from_binary_classification_data(
                10, elements_class_b, class_names
            )

    invalid_class_labels = [(1, "wrong"), ("label", -1), ("type", "!")]

    @pytest.mark.parametrize("class_labels", invalid_class_labels)
    def test_from_binary_classification_data_invalid_class_labels(self, class_labels):
        elements_class_b = ["other", "class", "elements"]
        class_names = ["class_a", "class_b"]

        with pytest.raises(TypeError):
            _ = DataBunch.from_binary_classification_data(
                10, elements_class_b, class_names, class_labels
            )


class TestTrainTestSplit:
    def test_init(self):
        tt_split = TrainTestSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
            name="sample",
        )

        assert tt_split.train_data
        assert tt_split.test_data
        assert tt_split.name == "sample"

    def test_default_name(self):
        tt_split = TrainTestSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
        )

        assert tt_split.name == "tt_split"

    def test_no_name_with_feature_info(self):
        tt_split = TrainTestSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
        )
        tt_split.add_feature_info("text")

        assert tt_split.name == "tt_split_text"

    def test_file_name_no_name(self):
        tt_split = TrainTestSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
        )

        assert tt_split.file_name() == "tt_split"

    def test_file_name_set_name(self):
        tt_split = TrainTestSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
            name="test",
        )

        assert tt_split.file_name() == "tt_split_test"

    def test_file_name_no_name_with_feature_info(self):
        tt_split = TrainTestSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
        )
        tt_split.add_feature_info("text")

        assert tt_split.file_name() == "tt_split_text"

    def test_create_info_dict(self):
        expected = {
            "train_data": {
                "shape": (3,),
                "feature_info": ["text"],
                "class_info": {
                    "num_positive_samples": 1,
                    "num_negative_samples": 2,
                    "positive_negative_ratio": 0.5,
                },
            },
            "test_data": {
                "shape": (3,),
                "feature_info": ["text"],
                "class_info": {
                    "num_positive_samples": 1,
                    "num_negative_samples": 2,
                    "positive_negative_ratio": 0.5,
                },
            },
            "feature_extractors": [],
            "name": "tt_split_text",
        }

        tt_split = TrainTestSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
        )
        tt_split.add_feature_info("text")

        assert tt_split.create_info_dict() == expected

    def test_stack_horizontally(self):
        tt_split = TrainTestSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
            name="sample",
        )
        other_tt_split = TrainTestSplit(
            DataBunch(np.array(["other", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["other", "testing", "data"]), np.array([0, 1, 0])),
            name="other_sample",
        )

        tt_split.stack_horizontally(other_tt_split)

        assert tt_split.train_data.shape == (3, 2)
        assert np.array_equal(
            tt_split.train_data.samples,
            np.array([["some", "other"], ["training", "training"], ["data", "data"]]),
        )
        assert np.array_equal(
            tt_split.test_data.samples,
            np.array([["some", "other"], ["testing", "testing"], ["data", "data"]]),
        )

    def test_stack_horizontally_invalid_type(self):
        tt_split = TrainTestSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
            name="sample",
        )

        ttv_split = TrainTestValidSplit(
            DataBunch(np.array(["other", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["other", "testing", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "validation", "data"]), np.array([0, 1, 0])),
            name="other_sample",
        )

        with pytest.raises(TypeError):
            tt_split.stack_horizontally(ttv_split)

    def test_to_valid_split(self):
        tt_split = TrainTestSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(
                np.array(["some", "testing", "data", "what"]), np.array([0, 1, 1, 0])
            ),
            name="sample",
        )

        ttv_split = tt_split.to_valid_split(seed=42)

        assert isinstance(ttv_split, TrainTestValidSplit)
        assert np.array_equal(tt_split.train_data.samples, ttv_split.train_data.samples)
        assert np.array_equal(tt_split.train_data.labels, ttv_split.train_data.labels)
        assert np.array_equal(ttv_split.test_data.samples, np.array(["data", "some"]))
        assert np.array_equal(ttv_split.test_data.labels, np.array([1, 0]))
        assert np.array_equal(
            ttv_split.validation_data.samples, np.array(["testing", "what"])
        )
        assert np.array_equal(ttv_split.validation_data.labels, np.array([1, 0]))


class TestTrainTestValidSplit:
    def test_init(self):
        ttv_split = TrainTestValidSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "validation", "data"]), np.array([0, 1, 0])),
            name="sample",
        )

        assert ttv_split.train_data
        assert ttv_split.test_data
        assert ttv_split.validation_data
        assert ttv_split.name == "sample"

    def test_default_name(self):
        ttv_split = TrainTestValidSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "validation", "data"]), np.array([0, 1, 0])),
        )

        assert ttv_split.name == "ttv_split"

    def test_no_name_with_feature_info(self):
        ttv_split = TrainTestValidSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "validation", "data"]), np.array([0, 1, 0])),
        )
        ttv_split.add_feature_info("text")

        assert ttv_split.name == "ttv_split_text"

    def test_file_name_no_name(self):
        ttv_split = TrainTestValidSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "validation", "data"]), np.array([0, 1, 0])),
        )

        assert ttv_split.file_name() == "ttv_split"

    def test_file_name_name_set(self):
        ttv_split = TrainTestValidSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "validation", "data"]), np.array([0, 1, 0])),
            name="test",
        )

        assert ttv_split.file_name() == "ttv_split_test"

    def test_file_name_no_name_with_feature_info(self):
        ttv_split = TrainTestValidSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "validation", "data"]), np.array([0, 1, 0])),
        )
        ttv_split.add_feature_info("text")

        assert ttv_split.file_name() == "ttv_split_text"

    def test_create_info_dict(self):
        expected = {
            "train_data": {
                "shape": (3,),
                "feature_info": ["text"],
                "class_info": {
                    "num_positive_samples": 1,
                    "num_negative_samples": 2,
                    "positive_negative_ratio": 0.5,
                },
            },
            "test_data": {
                "shape": (3,),
                "feature_info": ["text"],
                "class_info": {
                    "num_positive_samples": 1,
                    "num_negative_samples": 2,
                    "positive_negative_ratio": 0.5,
                },
            },
            "valid_data": {
                "shape": (3,),
                "feature_info": ["text"],
                "class_info": {
                    "num_positive_samples": 1,
                    "num_negative_samples": 2,
                    "positive_negative_ratio": 0.5,
                },
            },
            "feature_extractors": [],
            "name": "ttv_split_text",
        }

        ttv_split = TrainTestValidSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "validation", "data"]), np.array([0, 1, 0])),
        )
        ttv_split.add_feature_info("text")

        assert ttv_split.create_info_dict() == expected

    def test_stack_horizontally(self):
        ttv_split = TrainTestValidSplit(
            DataBunch(np.array(["some", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "testing", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["some", "validation", "data"]), np.array([0, 1, 0])),
            name="sample",
        )
        other_ttv_split = TrainTestValidSplit(
            DataBunch(np.array(["other", "training", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["other", "testing", "data"]), np.array([0, 1, 0])),
            DataBunch(np.array(["other", "validation", "data"]), np.array([0, 1, 0])),
            name="other_sample",
        )

        ttv_split.stack_horizontally(other_ttv_split)

        assert np.array_equal(
            ttv_split.train_data.samples,
            np.array([["some", "other"], ["training", "training"], ["data", "data"]]),
        )
        assert np.array_equal(
            ttv_split.test_data.samples,
            np.array([["some", "other"], ["testing", "testing"], ["data", "data"]]),
        )
        assert np.array_equal(
            ttv_split.validation_data.samples,
            np.array(
                [["some", "other"], ["validation", "validation"], ["data", "data"]]
            ),
        )
