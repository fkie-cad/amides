import numpy as np
import pytest
from sklearn.svm import SVC


from amides.data import TrainingResult, ValidationResult, MultiTrainingResult
from amides.data import DataBunch, TrainTestValidSplit


class TestTrainingResult:
    def test_default_name(self):
        train_rslt = TrainingResult(
            SVC(),
            TrainTestValidSplit(
                DataBunch(np.array(["other", "training", "data"]), np.array([0, 1, 0])),
                DataBunch(np.array(["other", "testing", "data"]), np.array([0, 1, 0])),
                DataBunch(
                    np.array(["other", "validation", "data"]), np.array([0, 1, 0])
                ),
            ),
        )

        assert train_rslt.name == "train_rslt_svc"

    def test_file_name_default_name(self):
        train_rslt = TrainingResult(
            SVC(),
            TrainTestValidSplit(
                DataBunch(np.array(["other", "training", "data"]), np.array([0, 1, 0])),
                DataBunch(np.array(["other", "testing", "data"]), np.array([0, 1, 0])),
                DataBunch(
                    np.array(["other", "validation", "data"]), np.array([0, 1, 0])
                ),
            ),
            timestamp="20220518_111030",
        )

        assert train_rslt.file_name() == "train_rslt_svc_20220518_111030"

    def test_file_name_no_default_name(self):
        train_rslt = TrainingResult(
            SVC(),
            TrainTestValidSplit(
                DataBunch(np.array(["other", "training", "data"]), np.array([0, 1, 0])),
                DataBunch(np.array(["other", "testing", "data"]), np.array([0, 1, 0])),
                DataBunch(
                    np.array(["other", "validation", "data"]), np.array([0, 1, 0])
                ),
            ),
            name="sample_result",
            timestamp="20220518_111030",
        )

        assert train_rslt.file_name() == "train_rslt_sample_result_20220518_111030"

    def test_create_info_dict(self):
        expected = {
            "estimator": "SVC",
            "estimator_params": SVC().get_params(),
            "data": {
                "train_data": {
                    "shape": (3,),
                    "feature_info": [],
                    "class_info": {
                        "num_positive_samples": 1,
                        "num_negative_samples": 2,
                        "positive_negative_ratio": 0.5,
                    },
                },
                "test_data": {
                    "shape": (3,),
                    "feature_info": [],
                    "class_info": {
                        "num_positive_samples": 1,
                        "num_negative_samples": 2,
                        "positive_negative_ratio": 0.5,
                    },
                },
                "valid_data": {
                    "shape": (3,),
                    "feature_info": [],
                    "class_info": {
                        "num_positive_samples": 1,
                        "num_negative_samples": 2,
                        "positive_negative_ratio": 0.5,
                    },
                },
                "feature_extractors": [],
                "name": "ttv_split",
            },
            "name": "sample_result",
            "timestamp": "20220518_111030",
        }

        train_rslt = TrainingResult(
            SVC(),
            TrainTestValidSplit(
                DataBunch(np.array(["other", "training", "data"]), np.array([0, 1, 0])),
                DataBunch(np.array(["other", "testing", "data"]), np.array([0, 1, 0])),
                DataBunch(
                    np.array(["other", "validation", "data"]), np.array([0, 1, 0])
                ),
            ),
            name="sample_result",
            timestamp="20220518_111030",
        )

        assert train_rslt.create_info_dict() == expected


class TestValidationResult:
    def test_default_name(self):
        valid_rslt = ValidationResult(
            SVC(),
            TrainTestValidSplit(
                DataBunch(np.array(["other", "training", "data"]), np.array([0, 1, 0])),
                DataBunch(np.array(["other", "testing", "data"]), np.array([0, 1, 0])),
                DataBunch(
                    np.array(["other", "validation", "data"]), np.array([0, 1, 0])
                ),
            ),
            np.array([1, 1, 0]),
        )

        assert valid_rslt.name == "valid_rslt_svc"

    def test_file_name_default_name(self):
        valid_rslt = ValidationResult(
            SVC(),
            TrainTestValidSplit(
                DataBunch(np.array(["other", "training", "data"]), np.array([0, 1, 0])),
                DataBunch(np.array(["other", "testing", "data"]), np.array([0, 1, 0])),
                DataBunch(
                    np.array(["other", "validation", "data"]), np.array([0, 1, 0])
                ),
            ),
            np.array([1, 1, 0]),
            timestamp="20220518_111030",
        )

        assert valid_rslt.file_name() == "valid_rslt_svc_20220518_111030"

    def test_file_name_no_default_name(self):
        valid_rslt = ValidationResult(
            SVC(),
            TrainTestValidSplit(
                DataBunch(np.array(["other", "training", "data"]), np.array([0, 1, 0])),
                DataBunch(np.array(["other", "testing", "data"]), np.array([0, 1, 0])),
                DataBunch(
                    np.array(["other", "validation", "data"]), np.array([0, 1, 0])
                ),
            ),
            np.array([1, 1, 0]),
            name="sample_result",
            timestamp="20220518_111030",
        )

        assert valid_rslt.file_name() == "valid_rslt_sample_result_20220518_111030"

    def test_create_info_dict(self):
        expected = {
            "estimator": "SVC",
            "estimator_params": SVC().get_params(),
            "data": {
                "train_data": {
                    "shape": (3,),
                    "feature_info": [],
                    "class_info": {
                        "num_positive_samples": 1,
                        "num_negative_samples": 2,
                        "positive_negative_ratio": 0.5,
                    },
                },
                "test_data": {
                    "shape": (3,),
                    "feature_info": [],
                    "class_info": {
                        "num_positive_samples": 1,
                        "num_negative_samples": 2,
                        "positive_negative_ratio": 0.5,
                    },
                },
                "valid_data": {
                    "shape": (3,),
                    "feature_info": [],
                    "class_info": {
                        "num_positive_samples": 1,
                        "num_negative_samples": 2,
                        "positive_negative_ratio": 0.5,
                    },
                },
                "feature_extractors": [],
                "name": "ttv_split",
            },
            "name": "sample_result",
            "timestamp": "20220518_111030",
        }

        valid_rslt = ValidationResult(
            SVC(),
            TrainTestValidSplit(
                DataBunch(np.array(["other", "training", "data"]), np.array([0, 1, 0])),
                DataBunch(np.array(["other", "testing", "data"]), np.array([0, 1, 0])),
                DataBunch(
                    np.array(["other", "validation", "data"]), np.array([0, 1, 0])
                ),
            ),
            np.array([1, 1, 0]),
            name="sample_result",
            timestamp="20220518_111030",
        )

        assert valid_rslt.create_info_dict() == expected


class TestMultiTrainingResult:
    def test_name_default(self):
        multi_rslt = MultiTrainingResult()

        assert multi_rslt.name == "multi_train_rslt"

    def test_file_name_default_name(self):
        multi_rslt = MultiTrainingResult(timestamp="20220518_120000")

        assert multi_rslt.file_name() == "multi_train_rslt_20220518_120000"

    def test_file_name_custom_name(self):
        multi_rslt = MultiTrainingResult(name="custom", timestamp="20220518_120000")

        assert multi_rslt.file_name() == "multi_train_rslt_custom_20220518_120000"

    def test_add_result(self):
        multi_rslt = MultiTrainingResult(name="custom", timestamp="20220518_120000")
        valid_rslt = ValidationResult(
            SVC(),
            TrainTestValidSplit(
                DataBunch(np.array(["other", "training", "data"]), np.array([0, 1, 0])),
                DataBunch(np.array(["other", "testing", "data"]), np.array([0, 1, 0])),
                DataBunch(
                    np.array(["other", "validation", "data"]), np.array([0, 1, 0])
                ),
            ),
            np.array([1, 1, 0]),
            name="sample_result",
            timestamp="20220518_111030",
        )

        multi_rslt.add_result(valid_rslt)

        assert len(multi_rslt.results.values()) == 1

    def test_create_info_dict(self):
        expected = {
            "name": "custom",
            "timestamp": "20220518_120000",
            "results": {
                "sample_result": {
                    "estimator": "SVC",
                    "estimator_params": SVC().get_params(),
                    "data": {
                        "train_data": {
                            "shape": (3,),
                            "feature_info": [],
                            "class_info": {
                                "num_positive_samples": 1,
                                "num_negative_samples": 2,
                                "positive_negative_ratio": 0.5,
                            },
                        },
                        "test_data": {
                            "shape": (3,),
                            "feature_info": [],
                            "class_info": {
                                "num_positive_samples": 1,
                                "num_negative_samples": 2,
                                "positive_negative_ratio": 0.5,
                            },
                        },
                        "valid_data": {
                            "shape": (3,),
                            "feature_info": [],
                            "class_info": {
                                "num_positive_samples": 1,
                                "num_negative_samples": 2,
                                "positive_negative_ratio": 0.5,
                            },
                        },
                        "feature_extractors": [],
                        "name": "ttv_split",
                    },
                    "name": "sample_result",
                    "timestamp": "20220518_111030",
                }
            },
        }

        multi_rslt = MultiTrainingResult(name="custom", timestamp="20220518_120000")
        valid_rslt = ValidationResult(
            SVC(),
            TrainTestValidSplit(
                DataBunch(np.array(["other", "training", "data"]), np.array([0, 1, 0])),
                DataBunch(np.array(["other", "testing", "data"]), np.array([0, 1, 0])),
                DataBunch(
                    np.array(["other", "validation", "data"]), np.array([0, 1, 0])
                ),
            ),
            np.array([1, 1, 0]),
            name="sample_result",
            timestamp="20220518_111030",
        )

        multi_rslt.add_result(valid_rslt)

        assert multi_rslt.create_info_dict() == expected
