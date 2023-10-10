"""This module contains components used for the visualization of 
classification results.

"""
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np

from abc import ABC, abstractmethod
from pandas import DataFrame, Series
from cycler import cycler
from matplotlib.legend import Legend

from sklearn.model_selection import validation_curve, learning_curve
from sklearn.calibration import calibration_curve

from amides.utils import get_current_timestamp


plt.style.use("seaborn-v0_8-colorblind")
sbn_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt_linestyles = ["-", "--", "-.", ":"]


class VisualizationError(BaseException):
    """Basic exception for all visualization-related errors."""


class Visualization(ABC):
    """Abstract base class for all further visualization classes."""

    def __init__(self):
        self._figure = None
        self._ax = None

    def show(self):
        plt.show()

    def save(self, output_path, **kwargs):
        self._figure.savefig(output_path, **kwargs)

    @abstractmethod
    def plot(self):
        pass


class DistributionPlot(Visualization):
    def __init__(self, x_label, y_label, data=None, name=None):
        super().__init__()
        self._name = name
        self._x_label = x_label
        self._y_label = y_label

        self._data = data if data is not None else []

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def file_name(self):
        return f"dist_{self._name}"

    def plot(self):
        self._figure, self._ax = plt.subplots()

        data = np.array(self._data)
        sum_data = data.sum()
        relative_data = data / sum_data

        x = [i for i in range(1, data.size + 1)]
        self._ax.bar(x, height=relative_data, color="blue")
        self._ax.set_xlim([0, 1])

        self._format_plot()

    def _format_plot(self):
        self._ax.set_xlabel(self._x_label)
        self._ax.set_ylabel(self._y_label)
        self._ax.grid(True)
        self._ax.set_title(self._name)
        self._figure.suptitle("Distribution")


class CumulativeDistributionPlot(Visualization):
    def __init__(self, x_label, y_label, data=None, name=None):
        super().__init__()
        self._name = name
        self._x_label = x_label
        self._y_label = y_label

        self._data = data if data is not None else []

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def file_name(self):
        return f"cum_dist_{self._name}"

    def plot(self):
        self._figure, self._ax = plt.subplots()

        data = np.array(self._data)
        sum_data = data.sum()
        y_sum = np.cumsum(self._data)
        y_relative = y_sum / sum_data

        x = [i for i in range(1, data.size + 1)]
        self._ax.plot(x, y_relative, color="blue")
        self._ax.fill_between(x, y_relative, alpha=0.5)
        self._ax.set_xlim([0, 1])
        self._ax.set_xlabel(self._x_label)
        self._ax.set_ylabel(self._y_label)
        self._ax.grid(True)
        self._ax.set_title(self._name)
        self._figure.suptitle("Cumulative Distribution")


class CombinedDistributionPlot(Visualization):
    def __init__(self, data=None, name=None):
        super().__init__()
        self._name = name

        self._data = data if data is not None else []
        self._figure = None
        self._ax = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def file_name(self):
        return f"combined_dist_{self._name}"

    def plot(self):
        self._figure, self._ax = plt.subplots(figsize=(6.4, 3.2))

        self._plot_data()
        self._format_plot()

    def _plot_data(self):
        data = np.array(self._data)
        sum_data = data.sum()
        y_relative = data / sum_data

        x = [i for i in range(1, data.size + 1)]
        self._ax.bar(x, height=y_relative, color=sbn_colors[2], width=1.0)
        self._ax.set_xlim([0, data.size + 1])

        y_sum = np.cumsum(self._data)
        y_sum_relative = y_sum / sum_data

        self._ax.plot(x, y_sum_relative, color=sbn_colors[0])
        self._ax.fill_between(x, y_sum_relative, alpha=0.1, color=sbn_colors[0])

    def _format_plot(self):
        self._ax.set_facecolor("#EBEBEB")
        for side in self._ax.spines:
            self._ax.spines[side].set_visible(False)

        self._set_x_ticks()
        self._ax.set_xlabel(
            "Attribution rank of true evaded rule", color="black", fontsize=13
        )
        self._ax.set_ylabel("Share of true alerts", color="black", fontsize=13)
        self._set_axvline()

        self._ax.tick_params(axis="both", which="major", labelsize=13)
        self._ax.grid(which="major", color="white", linewidth=1.2, axis="both")

        self._figure.tight_layout()

    def _set_x_ticks(self):
        start, end = self._ax.get_xlim()
        x_range = abs(end - start)

        if x_range <= 10:
            x_ticks = [i for i in range(int(start), int(end), 1)]
        elif x_range > 10 and x_range < 50:
            x_ticks = [i for i in range(int(start), int(end), 10)]
        elif x_range > 50:
            x_ticks = [i for i in range(int(start), int(end), 25)]

        self._ax.set_xticks(x_ticks)

    def _set_axvline(self):
        start, end = self._ax.get_xlim()
        x_range = abs(end - start)

        if x_range <= 5:
            return
        elif x_range > 5:
            ax_line_pos = 5
        elif x_range > 25:
            ax_line_pos = 10

        self._ax.axvline(ax_line_pos, 0, 1, color="black", linestyle="dashed")


class BoxPlot(Visualization):
    """BoxPlot-class to create box plots out of performance measurement data."""

    def __init__(self, x_label=None, y_label=None, name=None):
        super().__init__()
        self._x_label = x_label
        self._y_label = y_label
        self._name = name

        self._data = {}
        self._axs = []

    @property
    def y_label(self):
        return self._y_label

    @y_label.setter
    def y_label(self, y_label):
        self._y_label = y_label

    def set_data(self, data):
        """Add plot data which should be visualized. For each added PlotData instance, another
        box will be created.

        Parameter
        ---------
        data: np.ndarray
            Data which should be plotted.
        """
        self._data = DataFrame(
            dict([(key, Series(value)) for key, value in data.items()])
        )

    def plot(self):
        """Create  the plot."""
        self._figure, self._ax = plt.subplots()

        sbn.boxplot(data=self._data, ax=self._ax)

        self._format_plot()

    def _format_plot(self):
        self._create_seaborn_style_grid()
        self._ax.set_ylabel(self._y_label)
        self._ax.axhline(y=0.0)
        self._ax.axhline(y=0.72, linestyle="dashed")
        self._ax.set_ylim([-3, 3])

    def _create_seaborn_style_grid(self):
        self._ax.set_facecolor("#EBEBEB")
        for side in self._ax.spines:
            self._ax.spines[side].set_visible(False)

        self._ax.grid(which="major", color="white", linewidth=1.2)


class ViolinPlot(Visualization):
    """ViolinPlot-class to create violin plot out of performance measurement data."""

    def __init__(self, x_label=None, y_label=None, name=None):
        super().__init__()
        self._x_label = x_label
        self._y_label = y_label
        self._name = name

        self._data = None

    @property
    def y_label(self):
        return self._y_label

    @y_label.setter
    def y_label(self, y_label):
        self._y_label = y_label

    def set_data(self, data):
        """Add plot data which should be visualized. For each added PlotData instance, another violin
        will be created.

        Parameter
        ---------
        data: np.ndarray
            Data which should be plotted.
        """
        self._data = DataFrame(
            dict([(key, Series(value)) for key, value in data.items()])
        )

    def plot(self):
        """Create the plot."""
        self._figure, self._ax = plt.subplots(figsize=(6.4, 3.2))

        sbn.violinplot(data=self._data, ax=self._ax, zorder=10)
        self._format_plot()

    def _format_plot(self):
        self._create_seaborn_style_grid()
        self._ax.set_ylabel(self._y_label, fontsize=13)
        self._ax.axhline(y=0.57, linestyle="dashed", color="black")
        self._ax.set_ylim([-3.15, 3.15])
        self._ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        self._ax.tick_params(axis="both", which="major", labelsize=13)
        self._figure.tight_layout()

    def _create_seaborn_style_grid(self):
        self._ax.set_facecolor("#EBEBEB")

        for side in self._ax.spines:
            self._ax.spines[side].set_visible(False)

        self._ax.grid(which="both", color="white", linewidth=1.2)
        self._ax.set_axisbelow(True)


class SwarmPlot(Visualization):
    def __init__(self, x_label=None, y_label=None, name=None):
        super().__init__()
        self._x_label = x_label
        self._y_label = y_label
        self._name = name

        self._data = None

    @property
    def name(self):
        return self._name

    def set_data(self, data):
        """Add data which should be visualized. For each data array, another swarm
        will be created.

        Parameter
        ---------
        data: iterable
            Data which should be plotted.
        """
        self._data = DataFrame(data, columns=[self._x_label, self._y_label])

    def plot(self):
        self._figure, self._ax = plt.subplots()

        sbn.stripplot(
            x=self._x_label, y=self._y_label, data=self._data, ax=self._ax, size=5
        )
        sbn.boxplot(
            x=self._x_label,
            y=self._y_label,
            data=self._data,
            ax=self._ax,
            showcaps=False,
            boxprops={"facecolor": "None"},
            showfliers=False,
            whiskerprops={"linewidth": 0},
        )


class CalibrationCurve(Visualization):
    def __init__(self, num_bins=5, normalize=False, name=None):
        super().__init__()
        self._name = name
        self._num_bins = num_bins
        self._normalize = normalize

        self._data = []

    @property
    def name(self):
        return self._name

    def add_calibration_data(self, calibration_data):
        self._data.append(calibration_data)

    def plot(self):
        self._figure, self._ax = plt.subplots()

        for data in self._data:
            prob_true, prob_pred = calibration_curve(
                data.labels,
                data.probabilities,
                n_bins=self._num_bins,
                normalize=self._normalize,
            )
            self._ax.plot(prob_pred, prob_true, color="grey", linewidth=1)

        self._ax.plot([0, 1], [0, 1], color="red", linestyle="dashed")
        self._format_plot()

    def _format_plot(self):
        self._ax.grid()
        self._ax.set_xlabel("Mean predicted probability (Positive class: 1)")
        self._ax.set_ylabel("Fraction of positives (Positive class: 1)")
        self._ax.set_title(self._name)
        self._figure.suptitle("Calibration Plot")


class PrecisionRecallThresholdsPlot(Visualization):
    _colors = [sbn_colors[0], sbn_colors[2], sbn_colors[1]] + sbn_colors[3:]
    _linestyles = ["-", "--", ":", "-."]
    _num_result_metrics = 4

    def __init__(self, timestamp=None, name=None):
        super().__init__()
        self._name = name
        self._timestamp = (
            timestamp if timestamp is not None else get_current_timestamp()
        )

        self._evaluation_results = {}

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        self._timestamp = timestamp

    def file_name(self):
        if self._timestamp:
            file_name = f"prt_plot_{self._name}_{self._timestamp}"
        else:
            file_name = f"prt_plot_{self._name}"

        return file_name

    def add_evaluation_result(self, name, eval_result):
        """Adds given evaluation result. Raw data is used to
        calculate precision and recall data for the PR plot.

        Parameters
        ----------
        eval_result: EvaluationResult
            Evaluation results of validated estimator.
        """
        self._evaluation_results[name] = eval_result

    def plot(self):
        """Plots available PR data into a single diagram."""
        self._figure, self._ax = plt.subplots(figsize=(6.4, 3.2))
        self._init_prop_cycler()

        lines = []
        for result in self._evaluation_results.values():
            lines.append(self._plot_pr_from_evaluation_result(result))

        self._format_plot(lines)

    def _init_prop_cycler(self):
        colors, linestyles, alpha = [], [], []

        for i, _ in enumerate(self._evaluation_results.keys()):
            colors.extend([self._colors[i]] * self._num_result_metrics)
            linestyles.extend(self._linestyles)
            alpha.extend([1.0, 1.0, 0.3, 0.3])

        self._ax.set_prop_cycle(
            cycler(linestyle=linestyles) + cycler(color=colors) + cycler(alpha=alpha)
        )

    def _plot_pr_from_evaluation_result(self, result):
        precision, recall = self._plot_precision_recall_thresholds(
            result.precision, result.recall, result.thresholds
        )

        f1 = self._plot_f1_score(result.thresholds, result.f1_scores)
        mcc = self._plot_mcc_values(result.thresholds, result.mccs)

        return recall + precision + f1 + mcc

    def _mark_default_threshold(self, index):
        self._ax.axvline(index, 0, 1, color="#A0A0A0", linestyle="solid")

    def _plot_precision_recall_thresholds(self, precision, recall, thresholds):
        precision = self._ax.plot(thresholds, precision, label="Precision")
        recall = self._ax.plot(thresholds, recall, label="Recall")

        return precision, recall

    def _plot_f1_score(self, thresholds, f1_score):
        return self._ax.plot(thresholds, f1_score, label="F1")

    def _plot_mcc_values(self, thresholds, mcc):
        return self._ax.plot(thresholds, mcc, label="MCC")

    def _format_plot(self, lines):
        self._create_seaborn_style_grid()
        self._create_result_name_legend(lines)
        self._create_metrics_legend(lines)

        self._format_axes()

        if self._name is not None:
            self._ax.set_title(self._name)

        self._figure.tight_layout()

    def _create_seaborn_style_grid(self):
        self._ax.set_facecolor("#EBEBEB")
        for side in self._ax.spines:
            self._ax.spines[side].set_visible(False)

        self._ax.grid(which="major", color="white", linewidth=1.2)

    def _create_result_name_legend(self, lines):
        name_legend = Legend(
            self._ax,
            handles=self._get_result_name_handles(lines),
            labels=list(self._evaluation_results.keys()),
            fontsize=13,
            loc=(0.075, 0.25),
        )

        # rules_text = name_legend.get_texts()[0]
        # rules_text.set_usetex(True)
        # rules_text.set_text(r"\textsc{Amides}")

        self._ax.add_artist(name_legend)

    def _get_result_name_handles(self, lines: list[list]) -> list:
        handles = [result[1] for result in lines]

        return handles

    def _create_metrics_legend(self, lines: dict):
        metrics_legend = Legend(
            self._ax,
            handles=self._get_metrics_handles(lines),
            labels=["Recall", "Precision", "F1", "MCC"],
            loc=(0.72, 0.25),
            fontsize=13,
        )

        for handle in metrics_legend.legendHandles:
            handle.set_color("black")

        self._ax.add_artist(metrics_legend)

    def _get_metrics_handles(self, lines: list[list]) -> list:
        handles = [line for line in lines[0]]

        return handles

    def _format_axes(self):
        self._ax.set_xticks(
            np.arange(
                0,
                1.1,
                step=0.1,
            )
        )

        self._ax.set_xlabel("Decision Threshold", fontsize=13)
        self._ax.set_ylabel("Score", fontsize=13)
        self._ax.tick_params(axis="both", which="major", labelsize=13)


class MultiPRThresholdsPlot(PrecisionRecallThresholdsPlot):
    _colors = [sbn_colors[0], sbn_colors[2], sbn_colors[1]] + sbn_colors[3:]

    def __init__(self, name=None, timestamp=None):
        super().__init__(timestamp, name)

    def file_name(self):
        return f"multi_pr_plot_{self._name}_{self._timestamp}"

    def plot(self):
        self._figure, self._ax = plt.subplots(figsize=(6.4, 3.2))
        self._init_prop_cycler()

        result_lines = {}
        for name, result in self._evaluation_results.items():
            precision, recall = self._plot_pr_from_evaluation_result(result)
            result_lines[name] = [precision, recall]

        self._format_plot(result_lines)

    def _init_prop_cycler(self):
        linestyles, colors = [], []

        for i, _ in enumerate(self._evaluation_results.values()):
            linestyles.extend([plt_linestyles[0], plt_linestyles[1]])
            colors.extend([sbn_colors[i], sbn_colors[i]])

        self._prop_cycler = cycler(linestyle=linestyles) + cycler(color=colors)
        self._ax.set_prop_cycle(self._prop_cycler)

    def _plot_pr_from_evaluation_result(self, result):
        (precision,) = self._ax.plot(result.thresholds, result.precision)
        (recall,) = self._ax.plot(result.thresholds, result.recall)

        return precision, recall

    def _format_plot(self, lines):
        self._create_seaborn_style_grid()
        self._create_precision_recall_legend(lines)
        self._create_result_name_legend(lines)

        self._format_axes()

        if self._name is not None:
            self._ax.set_title(self._name)

        self._figure.tight_layout()

    def _create_seaborn_style_grid(self):
        self._ax.set_facecolor("#EBEBEB")
        for side in self._ax.spines:
            self._ax.spines[side].set_visible(False)

        self._ax.grid(which="major", color="white", linewidth=1.2)

    def _format_axes(self):
        self._ax.set_xticks(
            np.arange(
                0,
                1.1,
                step=0.1,
            )
        )

        self._ax.set_xlabel("Decision Threshold", fontsize=13)
        self._ax.set_ylabel("Score", fontsize=13)

        for label in self._ax.get_xticklabels():
            label.set_fontsize(13)

        for label in self._ax.get_yticklabels():
            label.set_fontsize(13)

    def _create_result_name_legend(self, lines):
        name_handles = self._get_result_name_handles(lines)
        name_legend = Legend(
            self._ax,
            handles=name_handles,
            labels=list(self._evaluation_results.keys()),
            fontsize=13,
            loc=(0.5, 0.2),
        )

        for handle in name_legend.legendHandles:
            handle.set_alpha(1.0)

        self._ax.add_artist(name_legend)

    def _get_result_name_handles(self, lines: dict) -> list:
        handles = [lines[key][1] for key in lines.keys()]

        return handles

    def _create_precision_recall_legend(self, lines: dict):
        precision_recall_handles = self._get_precision_recall_handles(lines)
        pr_legend = Legend(
            self._ax,
            handles=precision_recall_handles,
            labels=["Recall", "Precision"],
            loc=(0.05, 0.6),
            fontsize=13,
        )

        for handle in pr_legend.legendHandles:
            handle.set_color("black")

        self._ax.add_artist(pr_legend)

    def _get_precision_recall_handles(self, lines: dict) -> list:
        lines = next(iter(lines.values()))
        handles = [lines[1], lines[0]]

        return handles


class MultiTaintedPRThresholdsPlot(Visualization):
    _colors = [sbn_colors[0], sbn_colors[2], sbn_colors[1]] + sbn_colors[3:]

    def __init__(self, name=None, timestamp=None):
        super().__init__()
        self._name = name
        self._timestamp = timestamp

        self._evaluation_results = {}

    def file_name(self):
        if self._timestamp:
            file_name = f"multi_pr_plot_{self._name}_{self._timestamp}"
        else:
            file_name = f"multi_pr_plot_{self._name}"

        return file_name

    def add_evaluation_results(self, key, results):
        self._evaluation_results[key] = results

    def plot(self):
        self._figure, self._ax = plt.subplots(figsize=(6.4, 3.2))
        self._init_prop_cycler()

        result_lines = self._plot_pr_lines()

        self._format_plot(result_lines)

    def _init_prop_cycler(self):
        linestyles, colors, fades = [], [], []

        for i, results in enumerate(self._evaluation_results.values()):
            num_results = len(results)
            linestyles.extend([plt_linestyles[1], plt_linestyles[0]] * num_results)
            colors.extend([self._colors[i], self._colors[i]] * num_results)

            fades.extend([0.1, 0.1] * (num_results - 1))
            fades.extend([1.0, 1.0])

        self._prop_cycler = (
            cycler(linestyle=linestyles) + cycler(color=colors) + cycler(alpha=fades)
        )
        self._ax.set_prop_cycle(self._prop_cycler)

    def _plot_pr_lines(self):
        result_lines = {}
        for name, results in self._evaluation_results.items():
            result_lines[name] = []
            for result in results:
                precision, recall = self._plot_pr_from_evaluation_result(result)
                result_lines[name].append([recall, precision])

        return result_lines

    def _plot_pr_from_evaluation_result(self, result):
        (recall,) = self._ax.plot(result.thresholds, result.recall)
        (precision,) = self._ax.plot(result.thresholds, result.precision)

        return precision, recall

    def _format_plot(self, lines):
        self._create_seaborn_style_grid()
        self._create_tainted_degree_legend(lines)
        self._create_precision_recall_legend(lines)

        self._format_axes()

        if self._name is not None:
            self._ax.set_title(self._name)

        self._figure.tight_layout()

    def _create_seaborn_style_grid(self):
        self._ax.set_facecolor("#EBEBEB")
        for side in self._ax.spines:
            self._ax.spines[side].set_visible(False)

        self._ax.grid(which="major", color="white", linewidth=1.2)

    def _format_axes(self):
        self._ax.set_xticks(
            np.arange(
                0,
                1.1,
                step=0.1,
            )
        )

        self._ax.set_xlabel("Decision Threshold", fontsize=13)
        self._ax.set_ylabel("Score", fontsize=13)

        for label in self._ax.get_xticklabels():
            label.set_fontsize(13)

        for label in self._ax.get_yticklabels():
            label.set_fontsize(13)

    def _create_tainted_degree_legend(self, lines):
        td_handles = self._get_tainted_degree_handles(lines)
        td_legend = Legend(
            self._ax,
            handles=td_handles,
            labels=list(self._evaluation_results.keys()),
            fontsize=13,
            loc=(0.5, 0.1),
        )

        for handle in td_legend.legendHandles:
            handle.set_alpha(1.0)

        self._ax.add_artist(td_legend)

    def _get_tainted_degree_handles(self, lines: dict) -> list:
        handles = [lines[key][0][0] for key in lines.keys()]

        return handles

    def _create_precision_recall_legend(self, lines: dict):
        precision_recall_handles = self._get_precision_recall_handles(lines)
        pr_legend = Legend(
            self._ax,
            handles=precision_recall_handles,
            labels=["Recall", "Precision"],
            loc=(0.05, 0.1),
            fontsize=13,
        )

        for handle in pr_legend.legendHandles:
            handle.set_color("black")

        self._ax.add_artist(pr_legend)

    def _get_precision_recall_handles(self, lines: dict) -> list:
        lines = next(iter(lines.values()))
        handles = [lines[0][0], lines[0][1]]

        return handles


class WeightsFeaturesPlot(Visualization):
    def __init__(self, name, weights_features=None, num_top_features=20):
        super().__init__()
        self._name = name
        self._num_top_features = num_top_features

        self._weights_features = weights_features

    def set_weights_features(self, weights_features):
        self._weights_features = weights_features

    def plot(self):
        self._figure, self._ax = plt.subplots()

        colors = [
            "blue" if weight < 0 else "red" for weight in self._weights_features[:, 0]
        ]
        self._ax.barh(
            np.arange(len(self._weights_features[:, 0])),
            self._weights_features[:, 0],
            color=colors,
            align="edge",
        )

        self._ax.set_yticks(
            np.arange(len(self._weights_features[:, 0])),
            self._weights_features[:, 1],
            fontsize=12,
        )

        self._figure.tight_layout()
        plt.show()


class CumulativeDistanceDistributionPlot(Visualization):
    """CumulativeDistanceDistributionPlot to create cumulative distribution plot of
    hyperplane distances of true negative and false negative samples calculated
    by linear SVC model.
    """

    def __init__(self, tn_distances, fn_distances=None):
        """Create object instance.

        Parameters
        ----------
        tn_distances: List[number]
            List of distances of true negative (tn) samples.
        fn_distances: List[number]
            List of distances of false negative (fn) samples.

        """
        super().__init__()
        self._tn_distances = sorted(tn_distances)
        if fn_distances is not None:
            self._fn_distances = sorted(fn_distances)
        else:
            self._fn_distances = None

    @property
    def tn_distances(self):
        return self._tn_distances

    @tn_distances.setter
    def tn_distances(self, distances):
        self._tn_distances = sorted(distances)

    @property
    def fn_distances(self):
        return self._fn_distances

    @fn_distances.setter
    def fn_distances(self, distances):
        self._fn_distances = sorted(distances)

    def plot(self):
        if self._fn_distances is not None:
            self._plot_tn_fn_distances()
        else:
            self._plot_tn_distances()

    def save(self, output_path):
        self._figure.save(output_path, bbox_inches="tight")

    def _plot_tn_fn_distances(self):
        self._figure, self._axs = plt.subplots(2, 1)

        y_tn = [i for i in range(1, len(self._tn_distances) + 1)]
        y_fn = [i for i in range(1, len(self._fn_distances) + 1)]
        min_distance = min(min(self._tn_distances), min(self._fn_distances))
        max_distance = max(max(self._tn_distances), max(self._fn_distances))

        self._axs[0].set_xlim(min_distance, max_distance)
        self._axs[0].set_title("True Negatives (TNs)")
        self._axs[0].step(self._tn_distances, y_tn, where="post")
        self._axs[0].fill_between(self._tn_distances, y_tn, step="post", alpha=0.5)

        self._axs[1].set_xlim(min_distance, max_distance)
        self._axs[1].set_title("False Negatives (FNs)")
        self._axs[1].step(self._fn_distances, y_fn, where="post")
        self._axs[1].fill_between(self._fn_distances, y_fn, step="post", alpha=0.5)

    def _plot_tn_distances(self):
        self._figure, self._axs = plt.subplots(1, 1)

        y_tn = [i for i in range(1, len(self._tn_distances) + 1)]

        self._axs[0].set_title("True Negatives (TNs)")
        self._axs[0].step(self._tn_distances, y_tn, where="post")
        self._axs[0].fill_between(self._tn_distances, y_tn, step="post", alpha=0.5)


class ValidationCurve(Visualization):
    """ValidationCurve to plot training and test scores for varying parameter values. Find out bias and variance
    trade-off for a given parameter values range.
    """

    def __init__(
        self,
        estimator,
        data,
        labels,
        param_name,
        param_range,
        cv=None,
        scoring="accuracy",
    ):
        """Create ValidationCurve instance.

        Parameters
        ----------
        estimator: sklearn.base.BaseEstimator
            Estimator model that should be trained.
        data: np.ndarray
            Array holding the training and testing data (n_samples, n_features).
        labels: np.ndarray
            Corresponding labels (n_samples).
        param_name: str
            Name of parameter that should be varied throughought the optimization.
        param_range: np.ndarray (n_values)
            Parameter range that should be used for the optimization.
        cv: Optional[sklearn.model_selection.BaseSearchCV]
            Parameter optimization schema.
        scoring: str or callable.
            Either name or callable function.

        """
        super().__init__()
        self._estimator = estimator
        self._data = data
        self._labels = labels
        self._param_name = param_name
        self._param_range = param_range
        self._cv = cv
        self._scoring = scoring

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, cv):
        self._cv = cv

    @property
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, scoring):
        self._scoring = scoring

    def plot(self):
        """Calculate mean training and test scores and plot the results."""
        self._figure, self._ax = plt.subplots()
        (
            mean_train_scores,
            mean_test_scores,
        ) = self._calculate_mean_training_and_test_scores()

        self._ax.set_title(f"Validation Curve for {self._estimator.__class__.__name__}")
        self._ax.set_xlabel(f"{self._param_name}")
        self._ax.set_ylabel(
            "{0}".format(getattr(self._scoring, "__name__", repr(self._scoring)))
        )
        self._ax.set_ylim(0.0, 1.1)

        self._ax.grid()
        self._ax.plot(
            self._param_range,
            mean_train_scores,
            "o-",
            label="Mean Training Score",
            color="r",
        )
        self._ax.plot(self._param_range, mean_test_scores, "o-", label="CV Score")
        self._ax.legend(loc="best")

    def _calculate_mean_training_and_test_scores(self):
        train_scores, test_scores = validation_curve(
            self._estimator,
            X=self._data,
            y=self._labels,
            param_name=self._param_name,
            param_range=self._param_range,
            cv=self._cv,
            scoring=self._scoring,
        )

        mean_train_scores = np.mean(train_scores, axis=1)
        mean_test_scores = np.mean(test_scores, axis=1)

        return mean_train_scores, mean_test_scores


class LearningCurve(Visualization):
    """
    LearningCurve to plot cross-validated training and test scores for different training set sizes.
    Helps to find out how much we benefit from more training data and if model suffers from high variance or high bias
    (overfitting or underfitting).
    """

    def __init__(self, estimator, data, labels, cv=None, train_sizes=None):
        """Create visualization instance.

        Parameters
        ----------
        estimator: sklearn.base.BaseEstimator
            Model that should be trained.
        data: np.ndarray
            Array holding the training and test data (n_samples, n_features).
        labels: np.ndarray
            Labels of the training and testing data (n_samples).
        cv: Optional[sklearn.model_selection.BaseSearchCV]
            Parameter Optimizer.
        train_sizes: Optional[np.ndarray]
            Training sample sizes (n_ticks).

        """
        super().__init__()
        self._estimator = estimator
        self._data = data
        self._labels = labels
        self._cv = cv
        self._train_sizes = train_sizes

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, cv):
        self._cv = cv

    @property
    def train_sizes(self):
        return self._train_sizes

    @train_sizes.setter
    def train_sizes(self, train_sizes):
        self._train_sizes = train_sizes

    def plot(self):
        """Calculate mean training and test scores and create the plot."""
        self._figure, self._ax = plt.subplots()

        self._ax.set_title(f"Learning Curve for {self._estimator.__class__.__name__}")
        self._ax.set_xlabel("Training Samples")
        self._ax.set_ylabel("Score")

        (
            train_size_abs,
            mean_train_scores,
            mean_test_scores,
        ) = self._calculate_mean_train_and_test_scores()

        self._ax.grid()
        self._ax.plot(
            train_size_abs, mean_train_scores, "o-", color="r", label="Training Score"
        )
        self._ax.plot(train_size_abs, mean_test_scores, "o-", label="CV Score")
        self._ax.legend(loc="best")

    def _calculate_mean_train_and_test_scores(self):
        train_sizes_abs, train_scores, test_scores = learning_curve(
            self._estimator,
            self._data,
            self._labels,
            cv=self._cv,
            train_sizes=self._train_sizes,
        )

        mean_train_scores = np.mean(train_scores, axis=1)
        mean_test_scores = np.mean(test_scores, axis=1)

        return train_sizes_abs, mean_train_scores, mean_test_scores
