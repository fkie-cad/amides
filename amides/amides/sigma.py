import re
import os

from enum import Enum, auto
from luqum.parser import parser, ParseSyntaxError, IllegalCharacterError
from luqum.visitor import TreeVisitor
from luqum.tree import NoneItem

from amides.data import DataBunch, TrainTestValidSplit, TrainTestSplit
from amides.events import Events, EventsError
from amides.utils import (
    read_json_file,
    read_yaml_file,
    get_dir_names,
    get_logger,
    get_file_names,
)


_logger = get_logger(__name__)


class RuleDatasetError(BaseException):
    """RuleDataError for error related to rule specific data
    and operations.
    """

    def __init__(self, name, msg) -> None:
        super().__init__(f"Rule {name}: {msg}")


class RuleDataset:
    """
    RuleDataset class to hold matches, evasions, and rule filter of a single sigma rule.
    """

    match_regex = r"^.+_Match_\d+.json$"
    evasion_regex = r"^.+_Evasion_.+_\d+.json$"

    def __init__(self, name="rule_dataset", rule_filter=None):
        """Create RuleDataset instances.

        Parameters
        ----------
        name: str
            Name of the sigma rule.

        """
        self._name = name
        self._filter = rule_filter

        self._evasions = None
        self._matches = None

    @property
    def name(self):
        return self._name

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, rule_filter):
        self._filter = rule_filter

    @property
    def evasions(self):
        return self._evasions

    @evasions.setter
    def evasions(self, events):
        self._evasions = events

    @property
    def matches(self):
        return self._matches

    @matches.setter
    def matches(self, events):
        self._matches = events

    def create_matches_evasions_train_test_split(
        self, benign_train_events, benign_test_events
    ):
        """
        Creates training and test split out of benign, matching, and evasive
        events. Benign events are split in half.

        Training data: 1st half of benign events + matching events
        Test data: 2nd half of benign events + evasions

        Parameters
        ----------
        benign_train_events: Events
            Benign events for training data.
        benign_test_events: Events
            Benign events for test data.

        Returns
        -------
        train_test_split: Optional[TrainTestSplit]
            Training and Test data.

        """
        _logger.info("Creating matches-evasions train-test-split for '%s'", self._name)

        train_data = DataBunch.from_binary_classification_data(
            benign_train_events.data,
            self._matches.data,
            class_names=["benign", "matching"],
        )
        test_data = DataBunch.from_binary_classification_data(
            benign_test_events.data,
            self._evasions.data,
            class_names=["benign", "matching"],
        )
        train_test_split = TrainTestSplit(train_data, test_data, name=self._name)
        _logger.info("Created matches-evasions train-test-split for '%s'", self._name)

        return train_test_split

    def create_matches_evasions_validation_split(
        self,
        benign_train_events,
        benign_test_events,
        benign_valid_events,
        evasions_test_size=0.33,
        evasions_valid_size=0.66,
        evasions_split_seed=None,
    ):
        """Creates data split which can be used for calibration purposes. Benign
        events are split into three parts and distributed onto training, testing,
        and validation data.

        Returns
        -------
        valid_split: TrainTestValidSplit
            Training, testing, and validation data.
        """

        _logger.info(
            "Creating matches-evasions train-test-valid-split for %s", self._name
        )

        evasion_splits = self._split_evasions(
            split_sizes=[evasions_test_size, evasions_valid_size],
            seed=evasions_split_seed,
        )

        train_data = DataBunch.from_binary_classification_data(
            benign_train_events.data,
            self._matches.data,
            class_names=["benign", "matching"],
        )
        test_data = DataBunch.from_binary_classification_data(
            benign_test_events.data,
            evasion_splits[0].data,
            class_names=["benign", "matching"],
        )
        valid_data = DataBunch.from_binary_classification_data(
            benign_valid_events.data,
            evasion_splits[1].data,
            class_names=["benign", "matching"],
        )

        valid_split = TrainTestValidSplit(
            train_data, test_data, valid_data, name=self._name
        )
        _logger.info(
            "Created matches-evasions train-test-valid-split for %s", self._name
        )

        return valid_split

    def create_filter_evasions_train_test_split(
        self, benign_train_events, benign_test_events, search_fields
    ):
        """Creating training and testing split out of benign data and
        rule filters. Benign data is split according to benign_train_size
        and benign_test_size parameters.

        Returns
        -------
        train_test_split: TrainTestSplit
            Training and testing data.
        """
        _logger.info("Creating filter-evasions train-test-split for %s", self._name)

        wrapped_field_values = self.extract_field_values_from_filter(
            search_fields, wrap_up=True
        )

        train_data = DataBunch.from_binary_classification_data(
            benign_train_events.data,
            wrapped_field_values,
            class_names=["benign", "matching"],
        )
        test_data = DataBunch.from_binary_classification_data(
            benign_test_events.data,
            self._evasions.data,
            class_names=["benign", "matching"],
        )

        train_test_split = TrainTestSplit(train_data, test_data, name=self._name)
        _logger.info("Created filter-evasions train-test-split for %s", self._name)

        return train_test_split

    def create_filter_evasions_validation_split(
        self,
        benign_train_events,
        benign_test_events,
        benign_valid_events,
        search_fields,
        evasions_test_size=0.5,
        evasions_valid_size=0.5,
        evasions_split_seed=None,
    ):
        """Creats validation split which can be used in case classifier's
        class probabilities need to be calibrated.

        Returns
        -------
        calib_split: TrainTestCalibSplit
            Training, testing, and calibration data.
        """

        _logger.info(
            "Creating filter-evasions train-test-valid-split for '%s'", self._name
        )

        evasion_splits = self._split_evasions(
            split_sizes=[evasions_test_size, evasions_valid_size],
            seed=evasions_split_seed,
        )

        wrapped_field_values = self.extract_field_values_from_filter(
            search_fields, wrap_up=True
        )

        train_data = DataBunch.from_binary_classification_data(
            benign_train_events.data,
            wrapped_field_values,
            class_names=["benign", "matching"],
        )
        test_data = DataBunch.from_binary_classification_data(
            benign_test_events.data,
            evasion_splits[0].data,
            class_names=["benign", "matching"],
        )
        valid_data = DataBunch.from_binary_classification_data(
            benign_valid_events.data,
            evasion_splits[1].data,
            class_names=["benign", "matching"],
        )

        valid_split = TrainTestValidSplit(
            train_data, test_data, valid_data, name=self._name
        )
        _logger.info(
            "Created filter-evasions train-test-valid-split for '%s'", self._name
        )

        return valid_split

    def extract_field_values_from_filter(
        self, search_fields, wrap_up: bool = False
    ) -> list[str]:
        """
        Extracts all values of the specified search fields of the rule filter.

        Returns
        -------
        field_values: List[str]
            List of search field names whose values should be extracted

        """
        if not self._filter:
            raise RuleDatasetError(self._name, "No rule filter available")

        _logger.debug(
            "Extracting values for search fields %s from '%s'",
            search_fields,
            self._name,
        )

        field_values = []

        for rule_filter in self._filter:
            values = extract_field_values_from_filter(rule_filter, search_fields)
            field_values.extend(values)

        if wrap_up:
            field_values = self._wrap_up_field_values(search_fields, field_values)

        return field_values

    def load_events_and_filter(self, matches_evasions_path, rule_path):
        """
        Loads rule's event and filter data from the given events and rule paths.

        Parameters
        ----------
        matches_evasions_path: str
            Path of the directory holding matches and evasions.
        rule_path: str
            Path of the directory containing the rule filter.

        Raises
        ------
        EventsError
            If the specified events do not fit the requested type(s).
        RuleDataError
            In case no matches and evasions are available.

        """
        _logger.debug("Loading rule dataset for %s", self._name)
        self._load_matches_and_evasions(matches_evasions_path)
        self._load_rule_filter(rule_path)

    def _load_matches_and_evasions(self, matches_evasions_path):
        properties = self._load_properties(matches_evasions_path)
        event_type = self._get_required_event_type(properties)
        if event_type is None:
            raise RuleDatasetError(self._name, "No event type information available.")

        self._evasions = Events(event_type=event_type)
        self._matches = Events(event_type=event_type)

        _logger.debug(
            "Loading matches and evasions for %s from %s",
            self._name,
            matches_evasions_path,
        )

        self._insert_matches_and_evasions(matches_evasions_path)

        if self._matches.size == 0:
            _logger.debug("No matches for rule %s", self._name)

        if self._evasions.size == 0:
            _logger.debug("No evasions for rule %s", self._name)

    def _load_rule_filter(self, rule_path):
        try:
            _logger.debug("Loading rule filter from '%s'", rule_path)
            rules = read_yaml_file(rule_path)
            self._filter = self._extract_rule_filters(rules)
            self._name = self._extract_rule_name(rules)
        except (TypeError, IndexError) as err:
            raise RuleDatasetError(self._name, "No rule filter available") from err

    def _extract_rule_filters(self, rules):
        rule_filters = []

        for rule in rules:
            try:
                rule_filters.append(rule["filter"])
            except IndexError as err:
                _logger.error(err)
                continue

        return rule_filters

    def _extract_rule_name(self, rules):
        try:
            return rules[0]["pre_detector"]["title"]
        except (KeyError, IndexError):
            return "rule_dataset"

    def _load_properties(self, events_dir_path):
        properties_path = os.path.join(events_dir_path, "properties.yml")
        _logger.debug("Loading properties for %s from %s", self._name, properties_path)

        try:
            properties = read_yaml_file(properties_path)
            return properties[0]
        except IndexError as err:
            raise RuleDatasetError(self._name, "No properties.yml available") from err

    def _is_evasion_possible(self, properties):
        try:
            evasion_possible = properties["evasion_possible"]
            return not (evasion_possible == "unknown" or evasion_possible == "no")
        except KeyError as err:
            _logger.error(err)
            return False

    def _get_required_event_type(self, properties):
        try:
            event_type_name = properties["queried_event_types"][0]
            return Events.event_name_type_map[event_type_name]
        except (KeyError, ValueError) as err:
            _logger.error(err)
            return None

    def _load_event_from_file(self, events_dir_path, event_file_name):
        event_file_path = os.path.join(events_dir_path, event_file_name)
        _logger.debug("Loading event from %s", event_file_path)

        event = read_json_file(event_file_path)

        return event

    def _insert_matches_and_evasions(self, events_dir_path):
        event_file_names = self._get_events_file_names(events_dir_path)

        for event_file_name in event_file_names:
            event = self._load_event_from_file(events_dir_path, event_file_name)
            if event is not None:
                if self._is_match(event_file_name):
                    self._matches.add_event(event)
                elif self._is_evasion(event_file_name):
                    self._evasions.add_event(event)

    def _get_events_file_names(self, events_dir_path):
        try:
            file_names = get_file_names(events_dir_path)
            file_names.remove("properties.yml")

            return file_names
        except FileNotFoundError as err:
            raise RuleDatasetError(
                self._name, f"No matches or evasions in {events_dir_path}"
            ) from err
        except ValueError:
            # 'properties.yml' somehow disappeared, but this shouldn't bother us too much
            return file_names

    def _is_match(self, file_name):
        return re.search(RuleDataset.match_regex, file_name)

    def _is_evasion(self, file_name):
        return re.search(RuleDataset.evasion_regex, file_name)

    def _split_evasions(self, split_sizes, seed=None):
        splits = self._evasions.create_random_split(split_sizes=split_sizes, seed=seed)
        if not splits or len(splits) != len(split_sizes):
            raise RuleDatasetError(
                self._name,
                f"Could not split evasive events into {split_sizes} splits",
            )

        return splits

    def _wrap_up_field_values(self, search_fields, field_values):
        field_value_events = []

        for field_value in field_values:
            field_value_event = self._create_target_field_event(
                search_fields[0], field_value
            )
            field_value_events.append(field_value_event)

        return field_value_events

    def _create_target_field_event(self, search_field, field_value):
        fields = search_field.split(".")
        event = {}

        for field in fields[:-1]:
            event[field] = {}
            event = event[field]

        event[fields[-1]] = field_value

        return event


class RuleType(Enum):
    """RuleType enumeration for different rule set types."""

    WINDOWS_PROCESS_CREATION = auto()
    WINDOWS_REGISTRY_EVENT = auto()
    WINDOWS_POWERSHELL = auto()
    WEB_PROXY = auto()


class RuleSetDatasetError(BaseException):
    def __init__(self, rule_set_name, msg):
        super().__init__(f"RuleSetDataset {rule_set_name}: {msg}")


class RuleSetDataset:
    """
    RulesetDataset to hold event and rule data of a single sigma rule set family.
    """

    dir_name_rule_type_map = {
        "process_creation": RuleType.WINDOWS_PROCESS_CREATION,
        "registry_event": RuleType.WINDOWS_REGISTRY_EVENT,
        "web": RuleType.WEB_PROXY,
        "proxy": RuleType.WEB_PROXY,
    }

    def __init__(self, name=None, set_type=None):
        """Create rule set dataset instance and loads events (matches, evasions)
        and rules (rule-filter) from specified paths.

        Parameters
        ----------
        matches_evasions_path: str
            Path to the directory holding matches and evasions of the rule set.
        rules_path: str
            Path to the directory containing rule filters.
        name: str
            Name of the rule set.
        """

        self._name = name
        self._type = set_type

        self._rule_datasets = {}

    @property
    def rule_datasets(self):
        return self._rule_datasets

    @property
    def evasions(self):
        return self._get_evasions()

    @property
    def matches(self):
        return self._get_matches()

    def get_rule_dataset_by_name(self, rule_name):
        """
        Return specific rule dataset by its name.

        Parameters
        ----------
        rule_name: str
            Name of the rule.

        Returns
        -------
        rule_dataset: Optional[RuleDataset]
            RuleDataset or None.
        """
        return self._rule_datasets.get(rule_name)

    def create_matches_evasions_train_test_split(
        self, benign_train_events, benign_test_events
    ):
        """
        Create single training and test dataset out of all rule datasets.
        Benign data is split according to benign_train_size and benign_test_size.

        Returns
        -------
        train_test_data: Optional[TrainTestSplit]
            Training and test data.

        Raises
        ------
        RuleDataError
            In case no rule data is available

        DataError
            In case benign events could not be split in half.
        """
        _logger.info(
            "Creating matches-evasions train-test-split for rule-set %s", self._name
        )
        if not self._rule_datasets:
            raise RuleSetDatasetError(self._name, "No rule datasets available")

        matches = self._get_matches()
        evasions = self._get_evasions()

        train_data = DataBunch.from_binary_classification_data(
            benign_train_events.data,
            matches.data,
            class_names=["benign", "matching"],
        )
        test_data = DataBunch.from_binary_classification_data(
            benign_test_events.data,
            evasions.data,
            class_names=["benign", "matching"],
        )
        train_test_split = TrainTestSplit(train_data, test_data, name=self._name)
        _logger.info(
            "Created matches-evasions train-test-split for rule-set %s", self._name
        )

        return train_test_split

    def create_matches_evasions_validation_split(
        self,
        benign_train_events,
        benign_test_events,
        benign_valid_events,
        evasions_test_size=0.5,
        evasions_valid_size=0.5,
        evasions_split_seed=None,
    ):
        """Prepares TrainTestValidSplit containing training, testing, and
        validation data.

        Returns
        -------
        valid_split: TrainTestValidSplit
            Training, testing, and validation data.
        """
        _logger.info(
            "Creating matches-evasions train-test-valid-split for rule set %s",
            self._name,
        )
        if not self._rule_datasets:
            raise RuleSetDatasetError(self._name, "No rule datasets available")

        matches = self._get_matches()
        evasion_splits = self._split_evasions(
            split_sizes=[evasions_test_size, evasions_valid_size],
            seed=evasions_split_seed,
        )

        train_data = DataBunch.from_binary_classification_data(
            benign_train_events.data,
            matches.data,
            class_names=["benign", "matching"],
        )
        test_data = DataBunch.from_binary_classification_data(
            benign_test_events.data,
            evasion_splits[0].data,
            class_names=["benign", "matching"],
        )
        valid_data = DataBunch.from_binary_classification_data(
            benign_valid_events.data,
            evasion_splits[1].data,
            class_names=["benign", "matching"],
        )

        # Use 'process_creation' as rule name for an all-rules train and test split
        calib_split = TrainTestValidSplit(
            train_data, test_data, valid_data, name=self._name
        )
        _logger.info(
            "Created matches-evasions train-test-valid-split for rule-set %s",
            self._name,
        )

        return calib_split

    def create_filter_evasions_train_test_split(
        self, benign_train_events, benign_test_events, search_fields
    ):
        """Creates a single training and test split out of all rule datasets. Cmdline-arguments
           go into the training set, evasions go into the test set. Benign events are split
           according to benign_train_size and benign_test_size parameters.

        Parameters
        ----------
        seed: int
            Initial seeding value to recreate training and test split

        Returns
        -------
        train_test_data: TrainTestSplit
            Training and test data
        """

        _logger.info(
            "Creating filter-evasions train-test-split for rule set %s", self._name
        )
        if not self._rule_datasets:
            raise RuleSetDatasetError(self._name, "No rule datasets available")

        evasions = self._get_evasions()
        field_value_events = self.extract_field_values_from_filter(
            search_fields, wrap_up=True
        )

        train_data = DataBunch.from_binary_classification_data(
            benign_train_events.data,
            field_value_events,
            class_names=["benign", "matching"],
        )
        test_data = DataBunch.from_binary_classification_data(
            benign_test_events.data,
            evasions.data,
            class_names=["benign", "matching"],
        )
        # Use 'process_creation' as rule name for an all-rules train and test split
        train_test_split = TrainTestSplit(train_data, test_data, name=self._name)
        _logger.info(
            "Created filter-evasions train-test-split for rule set %s", self._name
        )

        return train_test_split

    def create_filter_evasions_validation_split(
        self,
        benign_train_events,
        benign_test_events,
        benign_valid_events,
        search_fields,
        evasions_test_size=0.5,
        evasions_valid_size=0.5,
        evasions_split_seed=None,
    ):
        """Creates a single calibration split out of all rule datasets. Rule
        filters go into the training set, evasions go into the test and
        validation set.

        Parameters
        ----------
        seed: int
            Initial seeding value to recreate training and test split

        Returns
        -------
        valid_data: TrainTestValidSplit
            Training, testing, and validation data.
        """

        _logger.debug(
            "Creating filter-evasions train-test-valid-split for rule set %s",
            self._name,
        )
        if not self._rule_datasets:
            raise RuleSetDatasetError(self._name, "No rule datasets available")

        evasion_splits = self._split_evasions(
            split_sizes=[evasions_test_size, evasions_valid_size],
            seed=evasions_split_seed,
        )
        field_value_events = self.extract_field_values_from_filter(
            search_fields, wrap_up=True
        )

        train_data = DataBunch.from_binary_classification_data(
            benign_train_events.data,
            field_value_events,
            class_names=["benign", "matching"],
        )
        test_data = DataBunch.from_binary_classification_data(
            benign_test_events.data,
            evasion_splits[0].data,
            class_names=["benign", "matching"],
        )
        valid_data = DataBunch.from_binary_classification_data(
            benign_valid_events.data,
            evasion_splits[1].data,
            class_names=["benign", "matching"],
        )
        valid_split = TrainTestValidSplit(
            train_data, test_data, valid_data, name=self._name
        )

        _logger.info(
            "Created filter-evasions train-test-valid-split for rule set %s", self._name
        )

        return valid_split

    def extract_field_values_from_filter(
        self, search_fields: list[str], wrap_up: bool = False
    ) -> list[str]:
        """
        Fetch search field values from filters of all rule datasets.

        Returns
        -------
        field_values: List[str]
            List of search field values.

        """
        field_values = []
        for rule_dataset in self._rule_datasets.values():
            try:
                rule_field_values = rule_dataset.extract_field_values_from_filter(
                    search_fields, wrap_up=wrap_up
                )
                field_values.extend(rule_field_values)
            except RuleDatasetError:
                continue

        _logger.debug(
            "Collected %d values for search fields from rule filters", len(field_values)
        )

        return field_values

    def load_rule_set_data(self, events_path, rules_path):
        """
        Load rules and corresponding event data from the specified directories.
        """
        _logger.debug("Loading rule set data")

        self._check_is_rule_set_events_dir(events_path)
        self._check_is_rule_set_rules_dir(rules_path)

        self._load_rules_data(events_path, rules_path)

    def _check_is_rule_set_events_dir(self, rule_set_events_dir):
        rule_set_type = self._get_rule_set_type_by_dir_name(rule_set_events_dir)
        if not rule_set_type:
            raise RuleSetDatasetError(
                self._name,
                f"Given rule set events dir {rule_set_events_dir} "
                f"does not correspond to any known rule set type",
            )

        if self._type:
            if self._type != rule_set_type:
                raise RuleSetDatasetError(
                    self._name,
                    f"Detected rule set type {rule_set_type} "
                    f"does not match specified type {self._type}",
                )
        else:
            self._type = rule_set_type

        dir_names = self._get_event_dir_names(rule_set_events_dir)
        if not dir_names:
            raise RuleSetDatasetError(
                self._name,
                f"Rule set dir {rule_set_events_dir}" f"does not contain any rule data",
            )

        return rule_set_type

    def _check_is_rule_set_rules_dir(self, rule_set_rules_dir):
        rule_set_type = self._get_rule_set_type_by_dir_name(rule_set_rules_dir)
        if not rule_set_type:
            raise RuleSetDatasetError(
                self._name,
                f"Given rule set rules dir {rule_set_rules_dir} "
                f"does not correspond to any known rule set type",
            )

        if self._type:
            if self._type != rule_set_type:
                raise RuleSetDatasetError(
                    self._name,
                    f"Detected rule set type {rule_set_type} "
                    f"does not match specified type {self._type}",
                )
        else:
            self._type = rule_set_type

        dir_names = self._get_rule_file_names(rule_set_rules_dir)
        if not dir_names:
            raise RuleSetDatasetError(
                self._name,
                f"Rule set dir {rule_set_rules_dir}" f"does not contain any rule data",
            )

    def _load_rules_data(self, rule_set_events_path, rule_set_rules_path):
        dir_names = self._get_event_dir_names(rule_set_events_path)
        for dir_name in dir_names:
            self._load_and_add_rule_data(
                dir_name, rule_set_events_path, rule_set_rules_path
            )

    def _get_event_dir_names(self, rule_set_events_dir):
        try:
            return get_dir_names(rule_set_events_dir)
        except FileNotFoundError as err:
            _logger.error(err)
            return []

    def _get_rule_file_names(self, rule_set_rules_dir):
        try:
            return get_file_names(rule_set_rules_dir)
        except FileNotFoundError as err:
            _logger.error(err)
            return []

    def _load_and_add_rule_data(
        self, rule_dir_name, rule_set_events_path, rule_set_rules_path
    ):
        try:
            rule_data = self._load_rule_data(
                rule_dir_name, rule_set_events_path, rule_set_rules_path
            )
            self._add_rule_dataset(rule_data)
        except (EventsError, RuleDatasetError) as err:
            _logger.error(err)

    def _load_rule_data(self, rule_dir_name, rule_set_events_path, rule_set_rules_path):
        rule_data_events_path = os.path.join(rule_set_events_path, rule_dir_name)
        rule_data_rule_path = os.path.join(rule_set_rules_path, f"{rule_dir_name}.yml")
        rule_data = RuleDataset(rule_dir_name, self._type)
        rule_data.load_events_and_filter(rule_data_events_path, rule_data_rule_path)

        return rule_data

    def _add_rule_dataset(self, rule_dataset):
        if rule_dataset.name in self._rule_datasets:
            raise RuleSetDatasetError(
                self._name, f"Rule dataset {rule_dataset.name} already in rule set"
            )

        self._rule_datasets[rule_dataset.name] = rule_dataset

    def _get_rule_set_type_by_dir_name(self, rule_set_dir):
        try:
            dir_name = os.path.basename(rule_set_dir)
            return self.dir_name_rule_type_map[dir_name]
        except KeyError:
            return None

    def _split_evasions(self, split_sizes, seed=None):
        evasive_events = self._get_evasions()
        splits = evasive_events.create_random_split(split_sizes=split_sizes, seed=seed)

        if not splits or len(split_sizes) != len(splits):
            raise RuleSetDatasetError(
                self._name, f"Could not split evasive events into {split_sizes} parts"
            )

        return splits

    def _get_matches(self):
        matches = []

        for rule_dataset in self._rule_datasets.values():
            matches.extend(rule_dataset.matches.data)

        event_type = self._get_rule_datasets_event_type()

        return Events(event_type, events=matches)

    def _get_evasions(self):
        evasions = []

        for rule_dataset in self._rule_datasets.values():
            evasions.extend(rule_dataset.evasions.data)

        event_type = self._get_rule_datasets_event_type()

        return Events(event_type, events=evasions)

    def _get_rule_datasets_event_type(self):
        return next(iter(self.rule_datasets.values())).matches.type


def extract_field_values_from_filter(rule_filter: str, fields: list[str]):
    """
    Extracts field values from rule filters.

    Parameters
    ----------
    rule_filter: str
        Rule filter possibly containing commandline arguments.

    Returns
    -------
    args: List[str]
        List of fields values from filter.

    """
    try:
        _logger.debug("Extracting field values '%s' from '%s'", fields, rule_filter)
        tree = parser.parse(rule_filter)
        visitor = MultiFieldVisitor(fields=fields)
        visitor.visit(tree)

        _logger.debug("Extracted values for fields '%s' : %s", fields, visitor.values)
        return visitor.values
    except (ParseSyntaxError, IllegalCharacterError) as err:
        _logger.error(err)
        return []


class MultiFieldVisitor(TreeVisitor):
    """
    MultiFieldVisitor to extract values of SearchFields in sigma rule filters.

    Cases where arguments need to be included:
        - SearchField whose name is in `fields`  and value is Phrase, RegEx(?)
        - SearchField whose name is in `fields` and value is FieldGroup
            - FieldGroup expressions consist of Word, Phrase, Regex, BaseApprox
        - SearchField whose name is in `fields` and has no parent which is a NotOperation
    """

    def __init__(self, fields):
        """Create visitor object."""
        super(MultiFieldVisitor, self).__init__(track_parents=False)
        self._fields = fields
        self._values = []

    @property
    def fields(self):
        return self._fields

    @property
    def values(self):
        return self._values

    def visit_search_field(self, node, context):
        """
        Visit SearchField objects. If name starts with one of the names in 'fields', visit child nodes.

        Parameters
        ----------
        node: luqum.tree.Item
            SearchFieldNode of the luqum tree.
        context: dict
            Context aka local parameters received from parent nodes.

        """
        match = False
        for field in self._fields:
            if node.name == field or node.name.startswith(field + "|"):
                match = True
        if match:
            context = self.child_context(node, NoneItem(), context)
            context[node.name.split("|")[0]] = True
        yield from self.generic_visit(node, context)

    def visit_phrase(self, node, context):
        """
        Visiting Phrase-objects. If Phrase is child of SearchField whose
        name starts with one of the names specified in 'fields', add its value to values..

        Parameters
        ----------
        node: luqum.tree.Item
            SearchFieldNode of the luqum tree.
        context: dict
            Context aka local parameters received from parent nodes.

        """
        for field in self._fields:
            if context.get(field, False):
                if node.value.startswith('"') and node.value.endswith('"'):
                    self._values.append(node.value[1:-1])
                else:
                    self._values.append(node.value)
        yield from self.generic_visit(node, context)

    def visit_not(self, node, context):
        yield NoneItem()
