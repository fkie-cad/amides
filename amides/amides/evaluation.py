import math
import numpy as np

from scipy import sparse
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix,
)

from amides.utils import get_current_timestamp


class BinaryEvaluationResult:
    """BinaryEvaluationResult to calculate and hold different metrics for
    the evaluation of binary classification tasks."""

    def __init__(self, thresholds=None, name=None, timestamp=None):
        self._thresholds = None
        self._precision = None
        self._recall = None
        self._f1_scores = None
        self._mccs = None
        self._no_skill = None

        self._tn, self._tp = None, None
        self._fn, self._fp = None, None

        if thresholds is not None:
            self._init_result_arrays(thresholds)

        self._name = name
        self._timestamp = (
            timestamp if timestamp is not None else get_current_timestamp()
        )

    @property
    def thresholds(self):
        return self._thresholds

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):
        self._precision = precision

    @property
    def recall(self):
        return self._recall

    @recall.setter
    def recall(self, recall):
        self._recall = recall

    @property
    def f1_scores(self):
        return self._f1_scores

    @property
    def mccs(self):
        return self._mccs

    @property
    def no_skill(self):
        return self._no_skill

    @property
    def name(self):
        if self._name is None:
            self._build_name_from_params()

        return self._name

    def evaluate(self, labels, predict):
        """Takes predicted values and labels and calculates evaluation metrics."""

        if self._thresholds is not None:
            self._evaluate_with_given_thresholds(labels, predict)
        else:
            self._evaluate_without_given_thresholds(labels, predict)

        self._calculate_no_skill(labels)

    def optimal_threshold_index(self):
        return np.argmax(self._f1_scores)

    def file_name(self):
        file_name = (
            self.name if self.name.startswith("eval_rslt") else f"eval_rslt_{self.name}"
        )

        if self._timestamp:
            file_name = f"{file_name}_{self._timestamp}"

        return file_name

    def create_info_dict(self):
        optimal_index = self.optimal_threshold_index()
        default_index = self.default_threshold_index()
        info = {
            "name": self.name,
            "timestamp": self._timestamp,
            "thresholds": {
                "num_thresholds": len(self._thresholds),
                "min_threshold": np.amin(self._thresholds),
                "max_threshold": np.amax(self._thresholds),
            },
            "optimum": {
                "threshold": self._thresholds[optimal_index],
                "f1_score": self._f1_scores[optimal_index],
                "precision": self._precision[optimal_index],
                "recall": self._recall[optimal_index],
                "mcc": self._mccs[optimal_index],
                "tn": self._tn[optimal_index],
                "tp": self._tp[optimal_index],
                "fn": self._fn[optimal_index],
                "fp": self._fp[optimal_index],
            },
            "default": {
                "threshold": self._thresholds[default_index],
                "f1_score": self._f1_scores[default_index],
                "precision": self._precision[default_index],
                "recall": self._recall[default_index],
                "mcc": self._mccs[default_index],
                "tn": self._tn[default_index],
                "tp": self._tp[default_index],
                "fn": self._fn[default_index],
                "fp": self._fp[default_index],
            },
        }

        return info

    def default_threshold_index(self):
        default_idx = np.where(self._thresholds == 0.5)

        if len(default_idx[0]) > 0:
            return default_idx[0][0]

        return int(len(self._thresholds) / 2)

    def _build_name_from_params(self):
        self._name = f"eval_rslt_{len(self._thresholds)}"

    def _init_result_arrays(self, thresholds):
        self._thresholds = thresholds
        self._num_thresholds = len(thresholds)

        if self._precision is None:
            self._precision = np.zeros(shape=(self._num_thresholds,))

        if self._recall is None:
            self._recall = np.zeros(shape=(self._num_thresholds,))

        self._f1_scores = np.zeros(shape=(self._num_thresholds,))
        self._mccs = np.zeros(shape=(self._num_thresholds,))
        self._tp = np.zeros(shape=(self._num_thresholds,))
        self._fp = np.zeros(shape=(self._num_thresholds,))
        self._tn = np.zeros(shape=(self._num_thresholds,))
        self._fn = np.zeros(shape=(self._num_thresholds,))

    def _evaluate_with_given_thresholds(self, labels, predict):
        for i, threshold in enumerate(self._thresholds):
            # print(f"Iteration: {i}")
            new_predict = np.where(predict >= threshold, 1, 0)
            self._precision[i] = precision_score(
                y_true=labels, y_pred=new_predict, zero_division=1
            )
            self._recall[i] = recall_score(y_true=labels, y_pred=new_predict)
            self._f1_scores[i] = f1_score(y_true=labels, y_pred=new_predict)
            self._mccs[i] = matthews_corrcoef(y_true=labels, y_pred=new_predict)
            self._tn[i], self._fp[i], self._fn[i], self._tp[i] = confusion_matrix(
                y_true=labels, y_pred=new_predict
            ).ravel()

    def _evaluate_without_given_thresholds(self, labels, predict):
        self._precision, self._recall, thresholds = precision_recall_curve(
            y_true=labels, probas_pred=predict
        )

        thresholds = np.append(thresholds, 1.0)
        self._init_result_arrays(thresholds)

        for i, threshold in enumerate(self._thresholds):
            new_predict = np.where(predict >= threshold, 1, 0)
            self._f1_scores[i] = f1_score(y_true=labels, y_pred=new_predict)
            self._mccs[i] = matthews_corrcoef(y_true=labels, y_pred=new_predict)
            self._tn[i], self._fp[i], self._fn[i], self._tp[i] = confusion_matrix(
                y_true=labels, y_pred=new_predict
            ).ravel()

    def _calculate_no_skill(self, labels):
        self._no_skill = len(labels[labels == 1]) / len(labels)


class RawBinaryEvaluationResult:
    """RawBinaryEvaluationResult to evaluate classification result against custom
    threshold values."""

    def __init__(self, thresholds, name=None, timestamp=None):
        """Create objects.

        Parameters
        ---------
        thresholds: iterable
            Iterable of threshold values. Values are used for classifer evaluation.
        name: Optional[str]
            Name of the evaluation result (Optional as it is mainly used for pickling)
        timestamp: Optional[str]
            Timestamp when result was created (Optional)
        """
        super().__init__(name, timestamp)
        self._thresholds = thresholds
        self._num_thresholds = len(thresholds)

        self._tp = np.zeros(shape=(self._num_thresholds,))
        self._fp = np.zeros(shape=(self._num_thresholds,))
        self._tn = np.zeros(shape=(self._num_thresholds,))
        self._fn = np.zeros(shape=(self._num_thresholds,))

    @property
    def name(self):
        if self._name is None:
            self._build_name_from_params()

        return self._name

    @property
    def tp(self):
        return self._tp

    @property
    def fp(self):
        return self._fp

    @property
    def tn(self):
        return self._tn

    @property
    def fn(self):
        return self._fn

    @property
    def thresholds(self):
        return self._thresholds

    def file_name(self):
        file_name = (
            self.name
            if self.name.startswith("th_eval_rslt")
            else f"th_eval_rslt_{self.name}"
        )

        return f"{file_name}_{self._timestamp}"

    def create_info_dict(self):
        optimal_index = self.optimal_threshold_index()
        precision = self._calculate_precision()
        recall = self._calculate_recall()
        f1_score = self._calculate_f_score()
        mcc = self._calculate_mcc()

        info = {
            "name": self.name,
            "timestamp": self._timestamp,
            "thresholds": {
                "num_thresholds": self._num_thresholds,
                "min_threshold_value": np.amin(self._thresholds),
                "max_threshold_value": np.amax(self._thresholds),
            },
            "max": {
                "f1_score": np.amax(f1_score),
                "precision": np.amax(precision),
                "recall": np.amax(recall),
                "mcc": np.amax(mcc),
            },
            "optimum": {
                "threshold": self._thresholds[optimal_index],
                "f1_score": f1_score[optimal_index],
                "precision": precision[optimal_index],
                "recall": recall[optimal_index],
                "mcc": mcc[optimal_index],
            },
        }

        return info

    def calculate_precision(self):
        precision = np.zeros(shape=(self._num_thresholds,))

        for i in range(0, self._num_thresholds):
            if self._tp[i] + self._fp[i] == 0:
                precision[i] = 1.0
            else:
                precision[i] = self._tp[i] / (self._tp[i] + self._fp[i])

        return precision

    def calculate_recall(self):
        recall = np.zeros(shape=(self._num_thresholds,))

        for i in range(0, self._num_thresholds):
            if self._tp[i] + self._fn[i] == 0:
                recall[i] = 1.0
            else:
                recall[i] = self._tp[i] / (self._tp[i] + self._fn[i])

        return recall

    def calculate_f_score(self):
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        f1_score = np.zeros(shape=(self._num_thresholds,))
        beta_squared = pow(1.0, 2)

        for i in range(0, self._num_thresholds):
            f1_score[i] = (1 + beta_squared) * (
                (precision[i] * recall[i]) / (beta_squared * precision[i] + recall[i])
            )

        return f1_score

    def _calculate_mcc(self):
        mcc = np.zeros(shape=(self._num_thresholds,))

        for i in range(0, self._num_thresholds):
            mcc[i] = (
                self._tp[i] * self._tn[i] - self._fp[i] * self._fn[i]
            ) / math.sqrt(
                (
                    (self._tp[i] + self._fp[i])
                    * (self._tp[i] + self._fn[i])
                    * (self._tn[i] + self._fp[i])
                    * (self._tn[i] + self._fn[i])
                )
            )

        return mcc

    def _calculate_no_skill(self):
        no_skill = (self._tp[0] + self._fn[0]) / (
            self._tp[0] + self._fp[0] + self._tn[0] + self._fn[0]
        )

        return no_skill

    def evaluate(self, labels, predict):
        for idx, threshold in enumerate(self._thresholds):
            adapted_predict = self._predict(threshold, predict)
            self._update_data(idx, adapted_predict, labels)

    def _predict(self, threshold, predict):
        return (predict[:] >= threshold).astype("int")

    def _update_data(self, threshold_idx, predict, labels):
        for i in range(0, len(predict)):
            if self._is_true_positive(labels[i], predict[i]):
                self._tp[threshold_idx] += 1
            elif self._is_true_negative(labels[i], predict[i]):
                self._tn[threshold_idx] += 1
            elif self._is_false_positive(labels[i], predict[i]):
                self._fp[threshold_idx] += 1
            elif self._is_false_negative(labels[i], predict[i]):
                self._fn[threshold_idx] += 1

    def _build_name_from_params(self):
        self._name = f"th_eval_rslt_{self._num_thresholds}"

    def _is_false_negative(self, label, predict):
        return label == 1 and predict == 0

    def _is_true_positive(self, label, predict):
        return label == 1 and predict == 1

    def _is_false_positive(self, label, predict):
        return label == 0 and predict == 1

    def _is_true_negative(self, label, predict):
        return label == 0 and predict == 0


class NegativeStateEvaluationResult(RawBinaryEvaluationResult):
    def __init__(self, origin_labels, thresholds, name=None, timestamp=None):
        super().__init__(thresholds, name, timestamp)
        self._init_negative_sample_state(origin_labels)
        self._num_positive_samples = 0

    def _init_negative_sample_state(self, origin_labels):
        negative_samples_end = self._get_negative_samples_end(origin_labels)
        self._negative_samples_state = np.zeros(
            shape=(
                self._num_thresholds,
                negative_samples_end,
            ),
            dtype="int64",
        )

    @property
    def negative_sample_state(self):
        return self._negative_samples_state

    def file_name(self):
        if self.name.startswith("state_eval_rslt"):
            file_name = self.name
        else:
            file_name = f"state_eval_rslt_{self.name}"

        return f"{file_name}_{self._timestamp}"

    def _calculate_no_skill(self):
        try:
            num_total_samples = (
                len(self._negative_samples_state[0]) + self._num_positive_samples
            )
        except IndexError:
            num_total_samples = self._num_positive_samples

        self._no_skill = self._num_positive_samples / num_total_samples

    def evaluate(self, predict, labels):
        self._update_num_positive_samples(labels)
        negative_samples_end = self._get_negative_samples_end(labels)

        for idx, threshold in enumerate(self._thresholds):
            adapted_predict = self._predict(threshold, predict)
            overall_predict = self._get_negative_samples_status(
                adapted_predict, negative_samples_end, idx
            )
            self._update_data(idx, overall_predict, labels)
            self._update_negative_samples_state(
                overall_predict, labels, negative_samples_end, idx
            )

    def _get_negative_samples_end(self, labels):
        return len([label for label in labels if label == 0])

    def _update_num_positive_samples(self, labels):
        self._num_positive_samples = +len(labels[labels == 1])

    def _get_negative_samples_status(self, predict, negative_samples_end, idx):
        predict[:negative_samples_end] = (
            self._negative_samples_state[idx] | predict[:negative_samples_end]
        ).astype(int)

        return predict

    def _update_negative_samples_state(
        self, predict, labels, negative_samples_end, idx
    ):
        self._negative_samples_state[idx] = (
            self._negative_samples_state[idx]
            | (
                (labels[:negative_samples_end] == 0)
                & (predict[:negative_samples_end] == 1)
            )
        ).astype(int)

    def _build_name_from_params(self):
        self._name = f"multi_eval_rslt_{self._metric}_{self._num_thresholds}"


class RuleAttributionEvaluationResult:
    def __init__(self, num_rules=130, name=None, timestamp=None):
        self._name = name
        self._timestamp = (
            timestamp if timestamp is not None else get_current_timestamp()
        )
        self._num_rules = num_rules

        self._num_total_samples = 0
        self._num_incomplete_samples = 0

        self._tp, self._fp = 0, 0
        self._tn, self._fn = 0, 0

        self._top_n_hits = np.zeros(shape=(self._num_rules,))
        self._top_n_hit_rates = np.zeros(shape=(self._num_rules,))

        self._misses = 0

    @property
    def name(self):
        if self._name is None:
            self._name = self._build_name_from_params()

        return self._name

    @property
    def tp(self):
        return self._tp

    @property
    def fp(self):
        return self._fp

    @property
    def tn(self):
        return self._tn

    @property
    def fn(self):
        return self._fn

    @property
    def num_total_samples(self):
        return self._num_total_samples

    @property
    def top_n_hits(self):
        return self._top_n_hits

    @property
    def top_n_hit_rates(self):
        return self._top_n_hit_rates

    @property
    def misses(self):
        return self._misses

    def file_name(self):
        file_name = (
            self.name
            if self.name.startswith("eval_rl_attr")
            else f"eval_rl_attr_{self.name}"
        )

        if self._timestamp:
            file_name = f"{file_name}_{self._timestamp}"

        return file_name

    def create_info_dict(self):
        info = {
            "name": self._name,
            "timestamp": self._timestamp,
            "num_rules": self._num_rules,
            "num_total_samples": self._num_total_samples,
            "num_incomplete_samples": self._num_incomplete_samples,
            "results": {
                "tp": self._tp,
                "fp": self._fp,
                "tn": self._tn,
                "fn": self._fn,
                "total_hits": np.sum(self._top_n_hits),
                "precision": self.calculate_precision(),
                "recall": self.calculate_recall(),
                "f1_score": self.calculate_f_score(),
                "top_1_hits": self._top_n_hits[0],
                "top_1_hit_rate": self._top_n_hit_rates[0],
                "top_10_hits": self._top_n_hits[:10].sum(),
                "top_10_hit_rate": self._top_n_hit_rates[:10].sum(),
            },
        }

        return info

    def calculate_precision(self):
        try:
            precision = self._tp / (self._tp + self._fp)
        except ZeroDivisionError:
            precision = 1.0

        return precision

    def calculate_recall(self):
        try:
            recall = self._tp / (self._tp + self._fn)
        except ZeroDivisionError:
            recall = 1.0

        return recall

    def calculate_no_skill(self):
        return (self._tp + self._fn) / (self._tp + self._fp + self._tn + self._fn)

    def calculate_f_score(self, beta=1):
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        beta_squared = pow(beta, 2)

        f_score = (1 + beta_squared) * (
            (precision * recall) / (beta_squared * precision + recall)
        )

        return f_score

    def calculate_top_n_hit_rates(self):
        self._top_n_hit_rates = self._top_n_hits / sum(self._top_n_hits)

    def calculate_miss_rate(self):
        return self._calculate_rate(self._misses, self._tp)

    def evaluate_rule_attributions(self, rule_name, rule_attributions):
        self._num_total_samples += 1

        if self._is_true_positive(rule_name, rule_attributions):
            self._tp += 1
            self._evaluate_rule_attributions(rule_name, rule_attributions)
        elif self._is_false_negative(rule_name, rule_attributions):
            self._fn += 1
        elif self._is_false_positive(rule_name, rule_attributions):
            self._fp += 1
        elif self._is_true_negative(rule_name, rule_attributions):
            self._tn += 1

    def _is_true_positive(self, rule_name, rule_attributions):
        return rule_name is not None and rule_attributions is not None

    def _is_false_negative(self, rule_name, rule_attributions):
        return rule_name is not None and rule_attributions is None

    def _is_false_positive(self, rule_name, rule_attributions):
        return rule_name is None and rule_attributions is not None

    def _is_true_negative(self, rule_name, rule_attributions):
        return rule_name is None and rule_attributions is None

    def _calculate_rate(self, num_hits, num_total):
        try:
            return num_hits / num_total
        except ZeroDivisionError:
            return num_hits

    def _evaluate_rule_attributions(self, rule_name, rule_attributions):
        sorted_rule_names = self._sort_rule_attributions(rule_attributions)

        try:
            for idx in range(self._num_rules):
                if rule_name in sorted_rule_names[: idx + 1]:
                    self._top_n_hits[idx] += 1
                    return
            # If we consider all rule confidence values, misses actually shouldn't be possible anymore
            self._misses += 1

        except IndexError:
            self._num_incomplete_samples += 1

    def _sort_rule_attributions(self, rule_attributions):
        sorted_rule_attributions = sorted(
            rule_attributions,
            key=lambda rl_attribution: rl_attribution[1],
            reverse=True,
        )

        sorted_rule_names = [rl_attrib[0] for rl_attrib in sorted_rule_attributions]

        return sorted_rule_names

    def _build_name_from_params(self):
        if self._timestamp:
            return f"eval_rl_attr_{self._timestamp}"

        return "eval_rl_attr"


def calculate_hyperplane_distances(svc, data):
    if isinstance(data, sparse.csr_matrix):
        return svc.decision_function(data) / sparse.linalg.norm(svc.coef_.copy())
    else:
        return svc.decision_function(data) / np.linalg.norm(svc.coef_.copy())
