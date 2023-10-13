import pytest
import numpy as np

from amides.evaluation import BinaryEvaluationResult


class TestBinaryClfEvaluationResult:
    def test_default_name(self):
        expected = "eval_rslt_dec_values_10"
        binary_result = BinaryClfEvaluationResult(
            np.linspace(0, 1, num=10), metric="dec_values"
        )

        assert binary_result.name == expected

    def test_file_name_default_name(self):
        expected = "eval_rslt_dec_values_10_19700101_000000"
        binary_result = BinaryClfEvaluationResult(
            np.linspace(0, 1, num=10), metric="dec_values", timestamp="19700101_000000"
        )

        assert binary_result.file_name() == expected

    def test_file_name_custom_name(self):
        expected = "eval_rslt_dec_values_10_19700101_000000"
        binary_result = BinaryClfEvaluationResult(
            np.linspace(0, 1, num=10),
            metric="dec_values",
            timestamp="19700101_000000",
            name="svc_rules",
        )

        assert binary_result.file_name() == "eval_rslt_svc_rules_19700101_000000"

    def test_create_info_dict(self):
        expected = {
            "name": "svc_rules",
            "timestamp": "19700101_000000",
            "metric": "dec_values",
            "thresholds": {
                "num_thresholds": 10,
                "min_threshold_value": 0.0,
                "max_threshold_value": 1.0,
            },
            "max_f1_score": 1.0,
            "max_precision": 1.0,
            "max_recall": 1.0,
            "optimal_threshold_value": 0,
        }
        binary_result = BinaryClfEvaluationResult(
            np.linspace(0, 1, num=10),
            metric="dec_values",
            timestamp="19700101_000000",
            name="svc_rules",
        )

        assert binary_result.create_info_dict() == expected
