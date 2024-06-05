"""This module provides classes and functions to hold and prepare datasets for the training and validation process.
"""

from abc import ABC, abstractmethod
import numpy as np

from scipy import sparse

from amides.utils import get_current_timestamp


class DataError(BaseException):
    """Base exception for all data-related errors."""


class DataBunch:
    """
    DataBunch to hold data points, their labels, and the label names.
    """

    def __init__(self, samples, labels, label_names=None, feature_info=None):
        """Create DataBunch instances.

        Parameters
        ---------
        samples: np.ndarray
            Array holding the data samples.
        labels: np.ndarray
            Samples' labels.
        label_names: Optional[List[str]]
            List of label names.
        feature_info: Optional[str]
            Feature information for data points of this bunch.

        Raises
        ------
        TypeError
            If  or labels are not of type np.ndarray.

        ValueError
            If size of data and labels are not equal.
        """
        if not (isinstance(samples, np.ndarray) or isinstance(samples, sparse.csr_matrix)):
            raise TypeError("samples is not of the required type np.ndarray or sparse.csr_matrix")

        if not isinstance(labels, np.ndarray):
            raise TypeError("'labels' is not of the required type 'np.ndarray'")

        if samples.shape[0] != labels.shape[0]:
            raise ValueError(
                "Number of data points in samples and number of labels" " should be equal"
            )

        self._samples = samples
        self._labels = labels

        if label_names is not None:
            self._label_names = label_names
        else:
            self._label_names = ["benign", "malicious"]

        if feature_info is not None:
            self._feature_info = feature_info
        else:
            self._feature_info = []

    @property
    def samples(self):
        """Return samples."""
        return self._samples

    @samples.setter
    def samples(self, samples):
        if not (isinstance(samples, np.ndarray) or isinstance(samples, sparse.csr_matrix)):
            raise TypeError("data is not of the required type np.ndarray or csr_matrix")

        self._samples = samples

    @property
    def labels(self):
        """Return labels."""
        return self._labels

    @labels.setter
    def labels(self, labels):
        if not isinstance(labels, np.ndarray):
            raise TypeError("labels is not of the required type np.ndarray")

        if labels.shape[0] != self._samples.shape[0]:
            raise ValueError("Number of labels and number of datapoints should be equal")

        self._labels = labels

    @property
    def label_names(self):
        """Return label names."""
        return self._label_names

    @label_names.setter
    def label_names(self, label_names):
        if not isinstance(label_names, list):
            raise TypeError("label_names is not of the required type list")

        self._label_names = label_names

    @property
    def feature_info(self):
        """Return feature info string."""
        return self._feature_info

    @property
    def size(self):
        """Return the number of samples in the bunch."""
        return self._samples.shape[0]

    @property
    def shape(self):
        """Return the shape of the samples array."""
        return self._samples.shape

    def add_feature_info(self, info):
        """Add feature info on data. Feature info is usually determined by
        transformations and feature extraction operations.

        Parameters
        ----------
        info: list
            Name of the contained feature type.

        """
        self._feature_info.append(info)

    def stack_horizontally(self, bunch):
        """
        Stacks data in data sequences horizontally (columnwise). Labels will
        remain untouched when performing this operation. Feature info will
        also be added.

        Parameters
        ----------
        bunch: DataBunch
            DataBunch whose data is stacked horizontally into this bunch.

        Raises
        ------
        TypeError
            If stacking data is not of the required type np.ndarray.

        ValueError
            If number of elements in stacking data does not fit.
        """
        if not isinstance(bunch, DataBunch):
            raise TypeError("bunch is not of type DataBunch")

        samples = bunch.samples

        if self._samples.shape[0] != samples.shape[0]:
            raise ValueError("Number of provided data points does not match" " number of samples")

        if self._samples.ndim == 1:
            self._samples = np.reshape(self._samples, newshape=(self._samples.shape[0], 1))

        if samples.ndim == 1:
            samples = np.reshape(samples, newshape=(samples.shape[0], 1))

        self._samples = np.hstack((self._samples, samples))

        self.add_feature_info(bunch.feature_info)

    def to_csr_matrix(self):
        """
        Turns n-dimensional data array into Compressed Sparse Row (CSR)-matrix.
        Saves a lot of space in case data is sparse.
        """
        self._samples = sparse.csr_matrix(self._samples)

    def split(self, num_splits=2, seed=None):
        """Split current data bunch into number of smaller data bunches. Resulting
        DataBunch-objects are equal in size and stratified.

        Parameters
        ----------
        num_splits: int
            Number of splits which should be created.
        seed: Optional[int]
            Seed random choice to recreate samples.

        Returns
        -------
        split_bunches: Tuple[DataBunch]
            Tuple holding created data splits.


        Raises
        ------
        ValueError
            If number of splits <= 2
            If number of positive samples < number of splits
            If number of negative samples < number of splits
        """
        if num_splits < 2:
            raise ValueError("Number of splits must be >= 2")

        positive_sample_indices = np.nonzero(self._labels)[0]
        negative_sample_indices = np.where(self._labels == 0)[0]

        if len(positive_sample_indices) < num_splits:
            raise ValueError(
                f"Number of elements with positive label "
                f"({len(positive_sample_indices)}"
                f" does not match number of splits ({num_splits})"
            )

        if len(negative_sample_indices) < num_splits:
            raise ValueError(
                f"Number of elements with negative label "
                f"({len(negative_sample_indices)})"
                f" does not match number of splits ({num_splits})"
            )

        if seed is not None:
            np.random.seed(seed)

        positive_sample_indices_splits = self._create_random_split(
            positive_sample_indices, num_splits
        )
        negative_sample_indices_splits = self._create_random_split(
            negative_sample_indices, num_splits
        )

        split_bunches = self._create_split_bunches(
            positive_sample_indices_splits, negative_sample_indices_splits
        )

        return tuple(split_bunches)

    def strip_elements_by_label(self, label):
        """Strip elements with specific label from data bunch. If label is not in labels,
           the array will remain unchanged.

        Parameters
        ----------
        label: int
            Integer label (0, 1)
        """
        label_indices = np.where(self._labels == label)[0]
        self._labels = np.delete(self._labels, label_indices)
        self._samples = np.delete(self._samples, label_indices, axis=0)

    def get_elements_by_label(self, label):
        """Returns all elements with specific label.

        Parameters
        ----------
        label: int
            Integer label (0, 1)
        """
        label_indices = np.where(self._labels == label)[0]

        return self._samples[label_indices]

    def create_info_dict(self):
        """Creates dictionary containing basic information on DataBunch instance."""
        info = {
            "shape": self._samples.shape,
            "feature_info": self._feature_info,
            "class_info": self._gather_class_info(),
        }

        return info

    def _create_split_bunches(self, positive_sample_indices_splits, negative_sample_indices_splits):
        split_bunches = []
        indices_splits = zip(positive_sample_indices_splits, negative_sample_indices_splits)

        for positive_sample_split, negative_sample_split in indices_splits:
            indices_split = np.concatenate([positive_sample_split, negative_sample_split])
            current_sample_split = np.take(self._samples, indices_split, axis=0)
            current_labels_split = np.take(self._labels, indices_split, axis=0)

            data_bunch = DataBunch(
                current_sample_split, current_labels_split, self._label_names, None
            )

            split_bunches.append(data_bunch)

        return split_bunches

    def _create_random_split(self, elements, num_splits):
        try:
            random_elements = np.random.choice(elements, size=len(elements), replace=False)
            splits = np.array_split(random_elements, num_splits, axis=0)
            return splits
        except ValueError:
            raise DataError(f"Could not split elements into {num_splits} parts") from ValueError

    def _gather_class_info(self):
        num_positive_samples = np.count_nonzero(self._labels == 1)
        num_negative_samples = np.count_nonzero(self._labels == 0)

        try:
            positive_negative_ratio = num_positive_samples / num_negative_samples
        except ZeroDivisionError:
            positive_negative_ratio = num_positive_samples

        class_info = {
            "num_positive_samples": num_positive_samples,
            "num_negative_samples": num_negative_samples,
            "positive_negative_ratio": positive_negative_ratio,
        }

        return class_info

    @classmethod
    def from_binary_classification_data(
        cls, elements_class_a, elements_class_b, class_names, class_labels=(0, 1)
    ):
        """
        Creates DataBunch out of binary classification data. Elements of the first
        class are labeled with '0' per default, labels of the second class are labeled
        with '1' per default. Optionaly, the classes' labels can be specified by the
        'class_labels' attribute.

        Parameters
        ----------
        elements_class_a: List[dict]
            List of elements representing the first class.
        elements_class_b: List[dict]
            List of elements representing the second class.
        class_names: List[str]
            List of class names
        class_labels: Tuple[int, int]
            Tuple containing labels for both classes.

        Returns
        -------
        data_bunch: DataBunch
            DataBunch created out of first class and second class elements..

        Raises
        ------
        TypeError
            In case any of the provided parameters are not of the required type
        """

        if not isinstance(elements_class_a, list):
            raise TypeError("elements_class_a is not of the required type list")

        if not isinstance(elements_class_b, list):
            raise TypeError("elements_class_b is not of the required type list")

        if not (isinstance(class_labels, tuple) and list(map(type, class_labels)) == [int, int]):
            raise TypeError("class_labels is not of the required type Tuple[int, int]")

        labels, samples = [], []

        samples.extend(elements_class_a)
        labels.extend(len(elements_class_a) * [class_labels[0]])
        samples.extend(elements_class_b)
        labels.extend(len(elements_class_b) * [class_labels[1]])

        return cls(np.array(samples), np.array(labels), class_names, None)


class DataSplit(ABC):
    """Base-class for all data which is splitted into multiple parts."""

    def __init__(self):
        self._data = {}

    @abstractmethod
    def file_name(self):
        """Returns file name which is mainly used when data splits should
        be pickled."""

    @abstractmethod
    def stack_horizontally(self, data_split):
        """Stack splitted data horizontally."""

    @abstractmethod
    def create_info_dict(self):
        """Return basic information on data split. Mainly used for integration
        when objects are being pickled."""


class TrainTestSplit(DataSplit):
    """
    TrainTestSplit to hold training and testing data for e.g. a specific rule
    or a set of rules.

    """

    def __init__(self, train_data=None, test_data=None, name=None):
        """Creates training and test data container object.

        Parameters
        ----------
        train_data: Optional[DataBunch]
            Training data.
        test_data: Optional[DataBunch]
            Test  data.
        name: str
            Name of the train-test split. Usually refers to the underlying type of data.

        """
        super().__init__()
        self._name = name
        self._data["train"] = train_data
        self._data["test"] = test_data

    @property
    def name(self):
        """(Sets) and returns the name of the split."""
        if self._name is None:
            self._build_name_from_data_info()

        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def train_data(self):
        """Return the training data."""
        return self._data["train"]

    @train_data.setter
    def train_data(self, data):
        self._data["train"] = data

    @property
    def test_data(self):
        """Return the test-data."""
        return self._data["test"]

    @test_data.setter
    def test_data(self, data):
        self._data["test"] = data

    def add_feature_info(self, info):
        """Add feature info in case any feature extraction or transformations
        were performed on both training and testing data.

        Parameters
        ----------
        info: str
            String describing feature extraction or transformation algorithm
            applied to data elements.
        """
        self._data["train"].add_feature_info(info)
        self._data["test"].add_feature_info(info)

    def to_valid_split(self, seed=None):
        """Split testing data into half to create additional validation split.

        Parameters
        ----------
        seed: Optional[int]
            Seeding to recreate calibration split

        Returns
        -------
        valid_split: TrainTestValidSplit
            Training, testing, and validation data
        """
        test_data, valid_data = self._data["test"].split(num_splits=2, seed=seed)
        valid_split = TrainTestValidSplit(
            self._data["train"], test_data, valid_data, name=self._name
        )

        return valid_split

    def file_name(self):
        """Creates file name. Mainly used when the training and testing data is
        about to be pickled.

        Returns
        -------
        file_name: str
            File name used when pickling objects.

        """
        file_name = self.name if self.name.startswith("tt_split") else f"tt_split_{self.name}"

        return file_name

    def stack_horizontally(self, data_split):
        """Stacks training data and testing data of given TrainTestSplit
        into this instance.

        Parameters
        ----------
        train_test_split: TrainTestSplit
            Training and test data which should be stacked onto this
            training and test data.

        Raises
        ------
        TypeError
            In case the provided object is not of type TrainTestSplit
        """
        if type(data_split) is not TrainTestSplit:
            raise TypeError("'data_split' is not of required type 'TrainTestSplit'")

        self._data["train"].stack_horizontally(data_split.train_data)
        self._data["test"].stack_horizontally(data_split.test_data)

    def create_info_dict(self):
        info = {
            "train_data": self._data["train"].create_info_dict() if self._data["train"] else None,
            "test_data": self._data["test"].create_info_dict() if self._data["test"] else None,
            "name": self.name,
        }

        return info

    def _build_name_from_data_info(self):
        self._name = "tt_split"

        if self._data["train"]:
            for info in self._data["train"].feature_info:
                self._name = f"{self._name}_{info}"


class TrainTestValidSplit(TrainTestSplit):
    """TrainTestValidSplit-class to create objects containing data splits
    for training, testing, and validation. Testing or validation data could also be used
    for other purposes.
    """

    def __init__(self, train_data=None, test_data=None, valid_data=None, name=None):
        """Initialize calibration split.

        Parameters
        ----------
        train_data: Optional[DataBunch]
            Training data
        test_data: Optional[DataBunch]
            Testing data
        valid_data: Optional[DataBunch]
            Validation data (or holdout dataset used for other purposes)
        name: Optional[str]
            Name of the data split.
        """
        super().__init__(train_data, test_data, name)
        self._data["valid"] = valid_data

    @property
    def validation_data(self):
        """Returns the validation data."""
        return self._data["valid"]

    @validation_data.setter
    def validation_data(self, data):
        self._data["valid"] = data

    def add_feature_info(self, info):
        super().add_feature_info(info)
        self._data["valid"].add_feature_info(info)

    def file_name(self):
        """Creates file name. Mainly used when the data split is pickled.

        Returns
        -------
        file_name: str
            File name used when pickling objects.

        """
        file_name = self.name if self.name.startswith("ttv_split") else f"ttv_split_{ self.name}"

        return file_name

    def stack_horizontally(self, data_split):
        """Stacks training, testing, and validation data of given
        TrainTestValidSplit onto data of this instance.

        Parameters
        ----------
        valid_split: TrainTestValidSplit
            The other validation data split

        Raises
        ------
        TypeError
            In case the provided data is not of the required TrainTestValidSplit
        """

        if type(data_split) is not TrainTestValidSplit:
            raise TypeError("data_split is not of required type TrainTestValidSplit")

        self._data["train"].stack_horizontally(data_split.train_data)
        self._data["test"].stack_horizontally(data_split.test_data)
        self._data["valid"].stack_horizontally(data_split.validation_data)

    def create_info_dict(self):
        info = {
            "train_data": self._data["train"].create_info_dict() if self._data["train"] else None,
            "test_data": self._data["test"].create_info_dict() if self._data["test"] else None,
            "valid_data": self._data["valid"].create_info_dict() if self._data["valid"] else None,
            "name": self.name,
        }

        return info

    def _build_name_from_data_info(self):
        self._name = "ttv_split"
        if self._data["train"]:
            for info in self._data["train"].feature_info:
                self._name = f"{self._name}_{info}"


class TrainingResult:
    """Holds trained estimator instance and the used training data."""

    def __init__(
        self,
        estimator,
        tainted_share=0.0,
        tainted_seed=0,
        data=None,
        scaler=None,
        feature_extractors=None,
        name=None,
        timestamp=None,
    ):
        """Init training results.

        Parameters
        ----------
        estimator: sklearn.base.BaseEstimator
            The trained estimator.
        data: DataBunch
            Data bunch holding training samples and labels used for estimator training.
        tainted_share: float
            Share of tainted training samples.
        tainted_seed: int
            Seeding used to create the tainted samples.
        feature_extractors: Optional[List[FeatureExtractor]]
            Feature extractors used to transform samples into feature vectors
        name: Optional[str]
            Name of the training result (Optional)
        timestamp: Optional[str]
            Timestamp when the results were created. If not explicitly specified,
            timestamp will be automatically generated.
        """
        self._estimator = estimator
        self._data = data
        self._tainted_share = tainted_share
        self._tainted_seed = tainted_seed
        self._scaler = scaler
        self._feature_extractors = feature_extractors if feature_extractors else []
        self._name = name
        self._timestamp = (
            timestamp if timestamp is not None else get_current_timestamp("%Y%m%d_%H%M%S")
        )

    @property
    def estimator(self):
        """Returns the trained model."""
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        self._estimator = estimator

    @property
    def data(self):
        """Returns the training data."""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def tainted_share(self):
        """Returns the fraction of tainting."""
        return self._tainted_share

    @tainted_share.setter
    def tainted_share(self, share):
        self._tainted_share = share

    @property
    def tainted_seed(self):
        """Returns the seeding used for tainting."""
        return self._tainted_seed

    @tainted_seed.setter
    def tainted_seed(self, seed):
        self._tainted_seed = seed

    @property
    def feature_extractors(self):
        """Returns the feature extractor."""
        return self._feature_extractors

    @property
    def scaler(self):
        """Returns the symmetric min-max scaler."""
        return self._scaler

    @scaler.setter
    def scaler(self, scaler):
        self._scaler = scaler

    @property
    def timestamp(self):
        """Returns the timestamp value."""
        return self._timestamp

    @property
    def name(self):
        """Returns the name of the result."""
        if self._name is None:
            self._build_name_from_result_info()

        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def add_feature_extractor(self, feat_extractor):
        """Add feature extractor to the result.

        Parameters
        ----------
        feat_extractor: Vectorizer
            The feature extractor to be added.
        """
        self._feature_extractors.append(feat_extractor)

    def file_name(self):
        """Creates file name string of training results. Mainly used for object
        pickling.

        Returns
        -------
        file_name: str
            The file name.

        """

        file_name = self.name if self.name.startswith("train_rslt") else f"train_rslt_{self._name}"

        if self._timestamp:
            file_name = f"{file_name}_{self.timestamp}"

        return file_name

    def create_info_dict(self):
        """Creates an info dict containin meta information in human-readable format.

        Returns
        -------
        :dict
            Dictionary containing meta information.
        """
        info = {
            "estimator": self._estimator.__class__.__name__,
            "estimator_params": self._estimator.get_params(),
            "data": self._data.create_info_dict() if self._data else None,
            "tainted_share": self._tainted_share,
            "tainted_seed": self._tainted_seed,
            "scaler": self._create_scaler_info() if self._scaler else None,
            "feature_extractors": (
                [extractor.__class__.__name__ for extractor in self._feature_extractors]
                if self._feature_extractors
                else None
            ),
            "name": self.name,
            "timestamp": self._timestamp,
        }

        return info

    def _create_scaler_info(self):
        return {
            "type": self._scaler.__class__.__name__,
            "data_min": self._scaler.data_min_,
            "data_max": self._scaler.data_max_,
            "range": self._scaler.data_range_,
            "scale": self._scaler.scale_,
            "min": self._scaler.min_,
        }

    def _build_name_from_result_info(self):
        self._name = f"train_rslt_{self._estimator.__class__.__name__.lower()}"


class ValidationResult(TrainingResult):
    """ValidationResult as extended TrainingResult to also hold predicition results of the (trained)
    estimator model.
    """

    def __init__(
        self,
        estimator,
        predict,
        tainted_share=0.0,
        tainted_seed=0,
        data=None,
        scaler=None,
        feature_extractors=None,
        name=None,
        timestamp=None,
    ):
        """Create ValidationResult.

        Parameters
        ----------
        estimator: sklearn.base.BaseEstimator
            The (trained) estimator.
        data: DataSplit
            Data split holding validation data used for validation.
        tainted_share: float
            Share of tainted training samples.
        tainted_seed: int
            Seeding used to create the tainted samples.
        predict: np.ndarray
            Prediction results generated out of test data and trained estimator.
        name: Optional[str]
            Name of the validation results
        timestamp: Optional[str]
            Timestamp when the results were created. If not given,
            timestamp will be automatically generated
        """
        super().__init__(
            estimator,
            tainted_share,
            tainted_seed,
            data,
            scaler,
            feature_extractors,
            name,
            timestamp,
        )
        self._predict = predict

    @property
    def predict(self):
        """Returns the decision function values."""
        return self._predict

    def file_name(self):
        file_name = self.name if self.name.startswith("valid_rslt") else f"valid_rslt_{self.name}"

        if self._timestamp:
            file_name = f"{file_name}_{self._timestamp}"

        return file_name

    def _build_name_from_result_info(self):
        self._name = f"valid_rslt_{self._estimator.__class__.__name__.lower()}"


class MultiTrainingResult:
    """MultiTrainingResult to hold multiple TrainingResult-instances, referenced by their name."""

    def __init__(self, name=None, timestamp=None, benign_training_data=None):
        """Create MultiTrainingResult instance.

        Parameters
        ----------
        name: Optional[str]
            Name of the result.
        timestamp: Optional[str]
            Timestamp when result data was created. In case none is provided,
            timestamp will be automatically set.
        """
        self._name = name
        self._timestamp = (
            timestamp if timestamp is not None else get_current_timestamp("%Y%m%d_%H%M%S")
        )

        self._results = {}
        self._benign_train_data = benign_training_data

    @property
    def name(self):
        """Return the name of the result."""
        if self._name is None:
            self._name = "multi_train_rslt"

        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def timestamp(self):
        """Return the timestamp value."""
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        self._timestamp = timestamp

    @property
    def results(self):
        """Return the results dictionary."""
        return self._results

    @property
    def benign_train_data(self):
        """Return the common benign training data."""
        return self._benign_train_data

    @benign_train_data.setter
    def benign_train_data(self, train_data):
        if not isinstance(train_data, DataBunch):
            raise TypeError("Provided benign training data is not of type DataBunch")

        self._benign_train_data = train_data

    def add_result(self, result):
        """Inserting result instance, using result name as key value."""
        if not isinstance(result, TrainingResult):
            raise TypeError("Provided object is not of type TrainingResult")

        if self._benign_train_data:
            result.data = self._remove_benign_samples(result.data)

        self._results[result.name] = result

    def get_result(self, result_name):
        """Fetching result instance by it's name."""
        try:
            result = self._results[result_name]
        except KeyError:
            return None

        return result

    def file_name(self):
        """Build a file name starting with 'multi_train_rslt'

        Returns
        -------
        :str
            The file name starting with 'multi_train_rslt'
        """
        if self.name.startswith("multi_train_rslt"):
            file_name = self.name
        else:
            file_name = f"multi_train_rslt_{self.name}"

        if self._timestamp:
            file_name = f"{file_name}_{self._timestamp}"

        return file_name

    def create_info_dict(self):
        """Creates an info dict containing meta information in human-readable format.

        Returns
        -------
        :dict
            Dictionary containing meta information.
        """
        results_info = {}

        for key, result in self._results.items():
            results_info[key] = result.create_info_dict()

        info = {
            "name": self.name,
            "timestamp": self._timestamp,
            "results": results_info,
        }

        return info

    def _rebuild_data(self, result, benign_data):
        try:
            feature_extractor = result.feature_extractors[0]
        except (IndexError, ValueError):
            return result

        benign_data = feature_extractor.transform(benign_data)
        result.data.stack_horizontally(benign_data)

        return result

    def _remove_benign_samples(self, data):
        data.strip_elements_by_label(0)

        return data


class MultiValidationResult(MultiTrainingResult):
    """MultiValidationResult to hold multiple ValidationResult-instances, referenced by the
    result's name.
    """

    def __init__(
        self,
        name=None,
        timestamp=None,
        benign_training_data=None,
        benign_validation_data=None,
    ):
        super().__init__(name, timestamp, benign_training_data)
        self._benign_valid_data = benign_validation_data

    @property
    def name(self):
        if self._name is None:
            self._name = "multi_valid_rslt"

        return self._name

    @property
    def benign_valid_data(self):
        """Returns common benign validation data."""
        return self._benign_valid_data

    @benign_valid_data.setter
    def benign_valid_data(self, data):
        if not isinstance(data, DataBunch):
            raise TypeError("Provided data is not of type DataBunch")

    def add_result(self, result):
        """Inserting result instance, using result name as key value."""
        if not isinstance(result, ValidationResult):
            raise TypeError("Provided object is not of type TrainingResult")

        if self._benign_train_data:
            result = self._remove_benign_samples(result.data.train_data)

        if self._benign_valid_data:
            result = self._remove_benign_samples(result.data.valid_data)

        self._results[result.name] = result

    def get_result(self, result_name):
        """Fetching result instance by it's name."""
        try:
            result = self._results[result_name]
        except KeyError:
            return None

        if self._benign_train_data:
            result = self._rebuild_data(result, self._benign_train_data)

        if self._benign_valid_data:
            result = self._rebuild_data(result, self._benign_valid_data)

        return result

    def file_name(self):
        if self.name.startswith("multi_valid_result"):
            file_name = self.name
        else:
            file_name = f"multi_valid_rslt_{self.name}"

        return f"{file_name}_{self._timestamp}"
