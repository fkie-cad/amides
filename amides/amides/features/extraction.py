"""This module contains functions and classes that are used for feature extraction."""

import numpy as np

from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

from amides.data import DataBunch
from amides.utils import get_current_timestamp, get_logger
from amides.features.tokenization import AnyWordCharacter, CommaSeparation
from amides.features.filter import NumericValues, Strings
from amides.features.preprocessing import FilterDummyCharacters, Lowercase

_logger = get_logger(__name__)


class TextFeatureExtractor(ABC):
    """Base class for all feature extractors that work on text data."""

    @property
    @abstractmethod
    def name(self):
        return

    @abstractmethod
    def file_name(self):
        return

    def extract(self, train_data, test_data=None, valid_data=None):
        """Convert training, test, and validation data into n-dimensional
        array of feature vectors.

        Parameters
        ----------
        train_data: DataBunch
            Training data (n_samples, n_features).
        test_data: DataBunch
            Test data (n_samples, n_features).
        valid_data: DataBunch
            Validation data (n_samples, n_features).

        Returns
        -------
        transformed_train_data: DataBunch
            Transformed training data.
        transformed_test_data: Optional[DataBunch]
            Transformed test data.
        transformed_valid_data: Optional[DataBunch]
            Transformed validation data.
        """
        train_data.samples = self.fit_transform(train_data.samples)
        train_data.add_feature_info(self.name)

        if test_data is not None:
            test_data.samples = self.transform(test_data.samples)
            test_data.add_feature_info(self.name)

        if valid_data is not None:
            valid_data.samples = self.transform(valid_data.samples)
            valid_data.add_feature_info(self.name)

        return train_data, test_data, valid_data

    @abstractmethod
    def fit_transform(self, samples):
        """Learn vocabulary from sample data and transform it into
        n-dimensional-array of feature vectors.

        Parameters
        ----------
        samples: np.ndarray
            Data samples.

        Returns
        -------
        transformed_samples: np.ndarray
            The transformed data samples.

        """
        return

    @abstractmethod
    def transform(self, samples):
        """
        Transform validation data into n-dimensional array of feature vectors.
        Use vocabulary learned from previous 'fit_transform'-call.

        Parameters
        ----------
        samples: np.ndarray
            Validation samples.

        Returns
        -------
        transformed_samples: np.ndarray
            Transformed data samples.

        """
        return


class TokenCountExtractor(TextFeatureExtractor):
    """TokenCountExtractor turns text data into matrix of token count vectors."""

    def __init__(
        self,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
    ):
        """Create TokenCountExtractor.

        Parameters
        ----------
        preprocessor: callable
            Preprocessor function to perform preprocessing of given input data.
        tokenizer: callable
            Function to extract tokens out of given text data.
        analyzer: str
            Determines if token counts should be made of word n-grams or
            character n-grams.
        ngram_range: tuple
            Lower and upper boundary for different word n-grams or
            character n-grams to be extracted.
        min_df: int
            Ignore tokens that have term frequency < min_df.
        max_df: int
            Ignore tokens that have term frequency > max_df.

        """
        super().__init__()
        self._vectorizer = CountVectorizer(
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )

    @property
    def name(self):
        return "token_count"

    @property
    def vectorizer(self):
        return self._vectorizer

    def file_name(self):
        return (
            f"{self.name}_{self._vectorizer.analyzer}_"
            f"{'_'.join(self._vectorizer.ngram_range)}"
            f"_{self._vectorizer.tokenizer.name}"
        )

    def fit_transform(self, samples):
        if not isinstance(samples, np.ndarray):
            raise TypeError("Given sample data is not of type np.ndarray")

        _logger.debug(
            "Extracting token count vectors from training data (n=%d)", samples.size
        )

        transformed_samples = self._vectorizer.fit_transform(samples).toarray()

        _logger.debug(
            "Extracted token count vectors from training data (shape=%s)",
            transformed_samples.size,
        )

        return transformed_samples

    def transform(self, samples):
        if not isinstance(samples, np.ndarray):
            raise TypeError("Given data samples are not of type np.ndarray")

        _logger.debug(
            "Extracting token count vectors from data samples (n=%d))", samples.size
        )

        transformed_samples = self._vectorizer.transform(samples).toarray()

        _logger.debug(
            "Extracted token count vectors from validation data (shape=%s)",
            transformed_samples.size,
        )

        return transformed_samples

    def get_feature_names(self):
        return self._vectorizer.get_feature_names_out()


class TfidfExtractor(TextFeatureExtractor):
    """Convert text data into n-dimensional darray of
    Term Frequency-Inverse Document Frequency (TF-IDF) vectors."""

    def __init__(
        self,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    ):
        """Create TfidfExtractor.

        Parameters
        ----------
        preprocessor: callable
            Preprocessor function to perform preprocessing on given input data.
        tokenizer: callable
            Function to extract tokens out of given text data.
        analyzer: str
            Determines if token counts should be made of word n-grams or
            character n-grams.
        ngram_range: tuple
            Lower and upper boundary for different word n-grams or
            character n-grams to be extracted.
        min_df: int
            Ignore tokens that have term frequency < min_df.
        max_df: int
            Ignore tokens that have term frequency > max_df.
        use_idf: Boolean
            Enable inverse-document-frequency-reweighting.
        smooth_idf: Boolean
            Prevents zero divisions.

        """
        super().__init__()
        self._vectorizer = TfidfVectorizer(
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
        )

    @property
    def name(self):
        return "tfidf"

    @property
    def vectorizer(self):
        return self._vectorizer

    def file_name(self):
        return (
            f"{self.name}_{self._vectorizer.analyzer}_"
            f"{'_'.join(map(str, self._vectorizer.ngram_range))}"
            f"_{self._vectorizer.tokenizer.name}"
        )

    def fit_transform(self, samples):
        if not isinstance(samples, np.ndarray):
            raise TypeError("Given samples is not of type np.ndarray")

        _logger.debug(
            "Extracting TF-IDF vectors from training data (n=%d)", samples.size
        )

        transformed_samples = self._vectorizer.fit_transform(samples).toarray()

        _logger.debug(
            "Extracted TF-IDF vectors from training data (shape=%s)",
            transformed_samples.size,
        )

        return transformed_samples

    def transform(self, samples):
        if not isinstance(samples, np.ndarray):
            raise TypeError("Given valid_data is not of type np.ndarray")

        _logger.debug(
            "Extracting TF-IDF vectors from data samples (n=%d))", samples.size
        )

        transformed_samples = self._vectorizer.transform(samples).toarray()
        _logger.debug(
            "Extracted TF-IDF vectors from validation data (shape=%s)",
            transformed_samples.size,
        )

        return transformed_samples

    def get_feature_names(self):
        return self._vectorizer.get_feature_names_out()


class LcsDistanceExtractor(TextFeatureExtractor):
    """This class allows to calculate the maximum Longest Common Substring (LCS) distance
    between text data and a set of reference sequences."""

    def __init__(self, reference_sequences):
        """Create LcsDistanceExtractor.

        Parameters
        ----------
        reference_sequences: List[str]
            Reference string sequences which are used as reference points to
            calculate the maximum LCS distance.

        Raises
        ------
        ValueError
            In case reference_sequences is no list of string sequences.

        """
        super().__init__()
        if not _is_valid_str_sequence_array(reference_sequences):
            raise TypeError(
                "'reference_data' is not of required type 'List[str]' or 'np.ndarray[str]'"
            )

        if isinstance(reference_sequences, list):
            reference_sequences = np.array(reference_sequences)

        self._reference_sequences = reference_sequences

    @property
    def name(self):
        return "lcs_distance"

    def file_name(self):
        return self.name

    def fit_transform(self, samples):
        if not isinstance(samples, np.ndarray):
            raise TypeError("Given samples is not of required type np.ndarray")

        _logger.debug(
            "Calculating maximum LCS distances between samples (n=%d}) "
            "and reference sequences (n=%d)",
            samples.size,
            self._reference_sequences.size,
        )

        distances = self.calculate_max_lcs_distances(samples)

        return distances

    def transform(self, samples):
        if not isinstance(samples, np.ndarray):
            raise TypeError("Given samples is not of required type np.ndarray")

        _logger.debug(
            "Calculating maximum LCS distances between validation data (n=%d) "
            "and reference data (n=%d)",
            samples.size,
            self._reference_sequences.size,
        )

        distances = self.calculate_max_lcs_distances(samples)

        return distances

    def calculate_max_lcs_distances(self, sequences):
        """
        Calculates the maximum LCS distances from sequence data to the
        reference data.

        Parameters
        ----------
        sequences: List[str]
            String sequences whose maximum LCS distance should be calculated.
        Returns
        -------
        distances: np.ndarray[int]
            Maximum LCS distances.

        Raises
        ------
        ValueError
            If provided input data is no list of character sequences.

        """
        if not _is_valid_str_sequence_array(sequences):
            raise TypeError(
                "Sequences are not of the required type List[str] or np.ndarray[str]"
            )

        if isinstance(sequences, list):
            sequences = np.array(sequences)

        distances = []
        for sequence in sequences:
            distances.append(
                max(
                    self.calculate_lcs_distance(reference_sequence, sequence)
                    for reference_sequence in self._reference_sequences
                )
            )

        return np.array(distances)

    def calculate_lcs_distance(self, sequence, other_sequence):
        """
        Calculates the Longest Common Substring (LCS) distance between two character sequences.

        Parameters
        ----------
        sequence: str
            First character sequence.
        other_sequence: str
            Second character sequence.

        Returns
        -------
        distance: int
            Cacluated LCS distance.

        """
        seq_matcher = SequenceMatcher(None, sequence, other_sequence)
        longest_match = seq_matcher.find_longest_match(
            0, len(sequence), 0, len(other_sequence)
        )

        return longest_match.size / len(sequence)


class RatcliffDistanceExtractor(TextFeatureExtractor):
    """This class allows to calculate the maximum Ratcliff-Obershelp distance
    between text data and a set of reference sequences."""

    def __init__(self, reference_sequences):
        """Create RatcliffDistanceExtractor.

        Parameters
        ----------
        reference_sequences: List[str]
            Reference character sequences which are used as reference points to
            calculate the maximum Ratcliff-Obershelp distance.

        Raises
        ------
        ValueError
            In case reference_sequences is no list of string sequences.

        """
        super().__init__()
        if not _is_valid_str_sequence_array(reference_sequences):
            raise TypeError(
                "Sequences are not of the required type 'List[str]' or 'np.ndarray[str]'"
            )

        if isinstance(reference_sequences, list):
            reference_sequences = np.array(reference_sequences)

        self._reference_sequences = reference_sequences

    @property
    def name(self):
        return "ratcliff_distance"

    def file_name(self):
        return self.name

    def fit_transform(self, samples):
        if not isinstance(samples, np.ndarray):
            raise TypeError("Given samples is not of required type np.ndarray")

        _logger.debug(
            "Calculating maximum Ratcliff distances between training data (n=%d) "
            "and reference data (n=%d)",
            samples.size,
            self._reference_sequences.size,
        )

        distances = self.calculate_max_ratcliff_distances(samples)

        return distances

    def transform(self, samples):
        if not isinstance(samples, np.ndarray):
            raise TypeError("Given data samples are not of required type DataBunch")

        _logger.debug(
            "Calculating maximum Ratcliff distances between validation data (n=%d) "
            "and reference data (n=%d)",
            samples.size,
            self._reference_sequences.size,
        )

        distances = self.calculate_max_ratcliff_distances(samples)

        return distances

    def calculate_max_ratcliff_distances(self, sequences):
        """
        Calculates the maximum Ratcliff-Obershelp distances between the given character
        sequences and the reference data.

        Parameters
        ----------
        sequences: List[str]
            List of character sequences.

        Returns
        -------
        distances: np.ndarray[int]
            Calculated distances.

        """
        if not _is_valid_str_sequence_array(sequences):
            raise TypeError(
                "'sequence_data' is of the required type 'List[str]' or 'np.ndarray[str]'"
            )

        if isinstance(sequences, list):
            sequences = np.array(sequences)

        distances = []
        for sequence in sequences:
            distances.append(
                max(
                    self.calculate_ratcliff_distance(reference_sequence, sequence)
                    for reference_sequence in self._reference_sequences
                )
            )

        return np.array(distances)

    def calculate_ratcliff_distance(self, sequence, other_sequence):
        """
        Calculates the Ratcliff-Obershelp distance between two character sequences.

        Parameters
        ----------
        sequence: str
            First character sequence.
        other_sequence: str
            Second character sequence.

        Returns
        -------
        distance: int
            Ratcliff-Obershelp distance.

        """
        seq_matcher = SequenceMatcher(None, sequence, other_sequence)

        return seq_matcher.ratio()


class ProcessArgsExtractor(TextFeatureExtractor):
    """This class performs extraction of the 'process.args'-field values of Sysmon
    Process Creation events.
    """

    def __init__(self):
        super().__init__()
        self._transformer = FunctionTransformer(
            ProcessArgsExtractor.extract_process_args
        )

    @property
    def name(self):
        return "process_args"

    def file_name(self):
        return self.name

    def extract(self, train_data, test_data=None, valid_data=None):
        """Convert training, test, and validation data into n-dimensional
        array of feature vectors.

        Parameters
        ----------
        train_data: DataBunch
            DataBunch of training data (n_samples, n_features).
        test_data: DataBunch
            DataBunch of testing data (n_samples, n_features).
        valid_data: DataBunch
            DataBunch of validation data (n_samples, n_features).

        Returns
        -------
        transformed_train_data: DataBunch
            The transformed training data.
        transformed_test_data: Optional[DataBunch]
            The transformed test data.
        transformed_valid_data: Optional[DataBunch]
            The transformed validation data.
        """
        transformed_data = self.fit_transform(train_data.samples)
        train_data = self._adjust_samples_labels_mismatch(train_data, transformed_data)
        train_data.add_feature_info(self.name)

        if test_data is not None:
            transformed_data = self.transform(test_data.samples)
            test_data = self._adjust_samples_labels_mismatch(
                test_data, transformed_data
            )
            test_data.add_feature_info(self.name)

        if valid_data is not None:
            transformed_data = self.transform(valid_data.samples)
            valid_data = self._adjust_samples_labels_mismatch(
                valid_data, transformed_data
            )
            valid_data.add_feature_info(self.name)

        return train_data, test_data, valid_data

    def fit_transform(self, samples):
        if not isinstance(samples, np.ndarray):
            raise TypeError("Given samples is not of required type np.ndarray")

        _logger.debug(
            "Extracting 'process.args' values from training data (n=%d)", samples.size
        )

        transformed_samples = self._transformer.fit_transform(samples)

        return transformed_samples

    def transform(self, samples):
        if not isinstance(samples, np.ndarray):
            raise TypeError("Given data samples are not of required type np.ndarray")

        _logger.debug(
            "Extracting 'process.args' values from samples (n=%d)", samples.size
        )

        transformed_samples = self._transformer.transform(samples)

        return transformed_samples

    @staticmethod
    def extract_process_args(events):
        """Extracts 'process.args' from Sysmon 'Process_Creation' events.

        Parameters
        ----------
        events: List[Dict]
            List of Sysmon 'Process_Creation' events

        Returns
        -------
        event_process_args: np.ndarray[str]
            Extracted 'process.args'-values

        """
        event_process_args = []
        for event in events:
            process_args = ProcessArgsExtractor.extract_process_args_from_event(event)
            event_process_args.append(process_args)

        return np.array(event_process_args)

    @staticmethod
    def extract_process_args_from_event(event):
        try:
            return event["process"]["args"]
        except KeyError:
            return None

    def _adjust_samples_labels_mismatch(self, data_bunch, transformed_samples):
        none_indices = np.where(transformed_samples == None)[0]
        adjusted_data = np.delete(transformed_samples, none_indices, axis=0)
        adjusted_labels = np.delete(data_bunch.labels, none_indices, axis=0)

        adjusted_bunch = DataBunch(
            adjusted_data,
            adjusted_labels,
            data_bunch.label_names,
            data_bunch.feature_info,
        )

        return adjusted_bunch


class CommandlineExtractor(TextFeatureExtractor):
    """This class performs extraction of commandline-values of Sysmon
    Process Creation events."""

    def __init__(self):
        super().__init__()
        self._transformer = FunctionTransformer(
            CommandlineExtractor.extract_commandline
        )

    @property
    def name(self):
        return "process_commandline"

    def file_name(self):
        return self.name

    def extract(self, train_data, test_data=None, valid_data=None):
        """Convert training, test, and validation data into n-dimensional
        array of feature vectors.

        Parameters
        ----------
        train_data: DataBunch
            DataBunch of training data (n_samples, n_features).
        test_data: DataBunch
            DataBunch of testing data (n_samples, n_features).
        valid_data: DataBunch
            DataBunch of validation data (n_samples, n_features).

        Returns
        -------
        transformed_train_data: DataBunch
            The transformed training data.
        transformed_test_data: Optional[DataBunch]
            The transformed test data.
        transformed_valid_data: Optional[DataBunch]
            The transformed validation data.
        """
        transformed_data = self.fit_transform(train_data.samples)
        train_data = self._adjust_data_labels_mismatch(train_data, transformed_data)
        train_data.add_feature_info(self.name)

        if test_data is not None:
            transformed_data = self.transform(test_data.samples)
            test_data = self._adjust_data_labels_mismatch(test_data, transformed_data)
            test_data.add_feature_info(self.name)

        if valid_data is not None:
            transformed_data = self.transform(valid_data.samples)
            valid_data = self._adjust_data_labels_mismatch(valid_data, transformed_data)
            valid_data.add_feature_info(self.name)

        return train_data, test_data, valid_data

    def fit_transform(self, samples):
        if not isinstance(samples, np.ndarray):
            raise TypeError("Given samples is not of required type np.ndarray")

        _logger.debug(
            "Extracting commandline-values from training data (n=%d)", samples.size
        )

        transformed_samples = self._transformer.fit_transform(samples)

        return transformed_samples

    def transform(self, samples):
        if not isinstance(samples, np.ndarray):
            raise TypeError("Given valid_data is not of required type np.ndarray")

        _logger.debug(
            "Extracting commandline-values from validation data (n=%d)", samples.size
        )

        transformed_samples = self._transformer.transform(samples)

        return transformed_samples

    @staticmethod
    def extract_commandline(events):
        """
        Extracts 'process.command_line' from Sysmon 'Process_Creation' events.


        Parameters
        ----------
        events: List[Dict]
            Sysmon 'Process_Creation' events.

        Returns
        -------
        events_commandline: np.ndarray[str]
            Extracted commandline-values.

        """
        events_commandline = []
        for event in events:
            proc_cmdline = CommandlineExtractor.extract_commandline_from_event(event)
            events_commandline.append(proc_cmdline)

        return np.array(events_commandline)

    @staticmethod
    def extract_commandline_from_event(event):
        proc_cmdline = CommandlineExtractor._extract_commandline_from_winlog(event)
        if proc_cmdline is None:
            proc_cmdline = CommandlineExtractor._extract_commandline(event)

        return proc_cmdline

    @staticmethod
    def _extract_commandline_from_winlog(event):
        try:
            return event["winlog"]["event_data"]["CommandLine"]
        except KeyError:
            return None

    @staticmethod
    def _extract_commandline(event):
        try:
            return event["process"]["command_line"]
        except KeyError:
            return None

    def _adjust_data_labels_mismatch(self, data_bunch, transformed_data):
        none_indices = np.where(transformed_data == None)[0]
        adjusted_data = np.delete(transformed_data, none_indices, axis=0)
        adjusted_labels = np.delete(data_bunch.labels, none_indices, axis=0)

        adjusted_bunch = DataBunch(
            adjusted_data,
            adjusted_labels,
            data_bunch.label_names,
            data_bunch.feature_info,
        )

        return adjusted_bunch



def _is_valid_str_sequence_array(seq_iter):
    """Checks if list or np.ndarray of strings is provided.

    Parameters
    ----------
    seq_list: List[str]
        Expected list of strings.

    Returns
    -------
    result: Boolean
        True/False.

    """
    if isinstance(seq_iter, (list, np.ndarray)):
        return all(isinstance(sequence, str) for sequence in seq_iter)

    return False
