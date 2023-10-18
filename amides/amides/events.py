import random

from enum import Enum, auto
from math import ceil

from amides.utils import get_file_paths, read_json_file, read_jsonl_file, get_logger


_logger = get_logger(__name__)


class EventsError(BaseException):
    """Errors related to events and their type."""


class EventType(Enum):
    """
    EventType Enum to enumerate different event types.
    """

    PROCESS_CREATION = auto()
    REGISTRY = auto()
    POWERSHELL = auto()
    PROXY_WEB = auto()


class Events:
    """
    Events class to hold multiple events of specific event type.
    """

    event_name_type_map = {
        "Microsoft-Windows-Sysmon_1": EventType.PROCESS_CREATION,
        "Microsoft-Windows-Sysmon_12": EventType.REGISTRY,
        "Microsoft-Windows-Sysmon_13": EventType.REGISTRY,
        "Microsoft-Windows-Powershell_4104": EventType.POWERSHELL,
        "Proxy-Web": EventType.PROXY_WEB,
    }

    event_type_name_map = {
        EventType.PROCESS_CREATION: "process_creation",
        EventType.REGISTRY: "registry_event",
        EventType.PROXY_WEB: "proxy",
        EventType.POWERSHELL: "powershell",
    }

    def __init__(self, event_type, name=None, events=None):
        """
        Init the events instance.

        Parameters
        ----------
        name: str
            Name of the events.
        event_type: EventType
            Type of the events.
        events: List[dict]
            List of the actual events.


        Raises
        -------
        TypeError
            In case 'events' is no list of events.
        EventTypeError
            If given events are not of the specified type.
        """
        self._name = name
        self._type = event_type
        self._data = []

        if events is not None:
            self.add_events(events)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def type(self):
        return self._type

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return len(self._data)

    @staticmethod
    def get_event_type_from_dir_name(dir_name):
        return Events.event_name_type_map[dir_name]

    def add_event(self, event):
        """
        Add single event to events.

        Parameters
        ----------
        event: dict
            Event which should be added.

        Raises
        ------
        TypeError
            If event is not of type Dict.

        EventTypeError
            If type of event and type of already added events do not match.
        """

        if type(event) is not dict:
            raise TypeError("'event' is not of required type Dict")

        self._add_event(event)

    def add_events(self, events):
        """Add multiple events.

        Parameters
        ----------
        events: list[dict]
            List of events which should be added.

        Raises
        ------
        TypeError
            If events is not of type List.

        EventTypeError
            If event type does not match type of these events.
        """

        if type(events) is not list:
            raise TypeError("Given 'events' is not of type List")

        for event in events:
            self.add_event(event)

    def load_from_dir(self, data_dir):
        """
        Load events from specified data directory.

        Parameters
        ----------
        data_dir: str
            Path to directory which holds events in JSON format.

        Raises
        ------
        FileNotFoundError
            In case the directory does not exist.
        PermissionError
            Insufficient privileges, file opened by other process, etc.
        """
        _logger.debug("Loading events from %s", data_dir)
        event_files = get_file_paths(data_dir)
        for event_file in event_files:
            self._load_events_from_file(event_file)

    def load_from_file(self, event_file):
        """Load events from specified event file.

        Parameters
        ----------
        event_file: str
            Path to file holding event(s) in JSON format.

        Raises
        ------
        FileNotFoundError
            In case the file does not exist.
        PermissionError
            Insufficient privileges, file opened by other process, etc.

        """
        self._load_events_from_file(event_file)

    def create_random_split(self, split_sizes, seed=None):
        """
        Splits events into random event splits according to
        givne splits list.


        Parameters
        ----------
        split_sizes: list
            List of split sizes.

        seed: int
            Seeding value to recreate the same event splits

        Returns
        -------
        splits: [List[dict]]
            List of lists with splitted events
        """

        self._check_split_sizes(split_sizes)

        _logger.debug(
            "Splitting %s events (num_events=%d) into %d splits (seed=%d)",
            self._type,
            self.size,
            len(split_sizes),
            seed,
        )

        splits = self._create_random_split(split_sizes, seed=seed)
        if splits:
            _logger.debug(
                "Number of splits: %d  Number of elements per split %d",
                len(split_sizes),
                splits[0].size,
            )

        return splits

    def _check_split_sizes(self, split_sizes):
        if not (
            isinstance(split_sizes, list)
            and len(split_sizes) > 1
            and all(isinstance(x, float) for x in split_sizes)
        ):
            raise TypeError("split_sizes is no list of float values")

        if not all(x > 0 for x in split_sizes):
            raise ValueError("split_sizes contains elements < 0")

        if sum(split_sizes) > 1.0:
            raise ValueError("Sum of split sizes > 1.0")

    def _load_events_from_file(self, event_file):
        if self._is_json_file(event_file):
            self._load_event_from_json_file(event_file)
        elif self._is_jsonl_file(event_file):
            self._load_events_from_jsonl_file(event_file)

    def _load_event_from_json_file(self, event_file):
        _logger.debug("Loading event from %s", event_file)
        event = read_json_file(event_file)

        try:
            self._add_event(event)
        except EventsError as err:
            _logger.error(err)

    def _load_events_from_jsonl_file(self, events_file):
        _logger.debug("Loading events from %s", events_file)
        events = read_jsonl_file(events_file)
        for event in events:
            try:
                self._add_event(event)
            except EventsError as err:
                _logger.error(err)
                continue

    def _add_event(self, event):
        if self._is_required_type_of_event(event):
            self._data.append(event)
            _logger.debug("Adding event %s to %s events", event, self._type)
        else:
            raise EventsError(
                f"Event type does not match required event type {self._type}"
            )

    def _is_json_file(self, event_file):
        return event_file.endswith(".json")

    def _is_jsonl_file(self, event_file):
        return event_file.endswith(".jsonl")

    def _is_required_type_of_event(self, event):
        return True

    def _create_random_split(self, split_sizes, seed=None):
        if seed is not None:
            random.seed(seed)

        if not self._data:
            return []

        try:
            random_events = random.sample(self._data, k=self.size)
            splits = []
            split_idx = 0

            for split_size in split_sizes:
                absolute_split_size = ceil(self.size * split_size)
                current_split = random_events[
                    split_idx : split_idx + absolute_split_size
                ]
                split_idx += absolute_split_size

                events = Events(self._type, events=current_split, name=self._name)
                splits.append(events)

            return splits
        except ValueError as err:
            _logger.error(err)
            return []


class EventsCache:
    """
    EventsCache to cache events of specific event type in order to be re-used by e.g.
    RuleDataset objects.
    """

    def __init__(self):
        """Create events cache."""
        self._events = {}

    @property
    def events(self):
        return self._events

    def add_events(self, events):
        """Add events of a specific event type.

        Parameters
        ----------
        events: Events
            Events of specific type which should be added.

        Raises
        ------
        EventTypeError
            In case events of the same type are already part of the cache.
        """
        if events.name in self._events:
            raise EventsError(
                f"Events with name {events.name} are already in the cache"
            )

        self._events[events.name] = events

    def get_events_by_name(self, name):
        """Get cached events with specific name.

        Parameters
        ----------
        name: str
            Name of the requested events.

        Returns
        -------
        events: Events
            Events of the requested type.
        """
        return self._events.get(name)

    def get_events_by_type(self, event_type):
        """Get cached events of specific event type.

        Parameters
        ----------
        event_type: EventType
            Requested event type.

        Returns
        -------
        events: List[Events]
            List of events of the requested type.
        """

        events_list = []

        for events in self._events.values():
            if events.type == event_type:
                events_list.append(events)

        return events_list


# Global events cache to hold once loaded benign event data
benign_events_cache = EventsCache()
