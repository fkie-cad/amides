import pytest
import os

from amides.events import (
    Events,
    EventsCache,
    EventsError,
    EventType,
)


def data_path():
    return os.path.realpath(os.path.join(os.path.dirname(__file__), "../data"))


def pc_events_json_path():
    return os.path.join(data_path(), "socbed-sample/process_creation/json")


def pc_events_jsonl_path():
    return os.path.join(data_path(), "socbed-sample/process_creation/jsonl")


def powershell_events_jsonl_path():
    return os.path.join(data_path(), "socbed-sample/powershell/jsonl")


def sigma_path():
    return os.path.join(data_path(), "sigma-study")


class TestEvents:
    def test_init(self):
        assert Events("process_creation", EventType.PROCESS_CREATION)
        assert Events("powershell", EventType.POWERSHELL)
        assert Events("registry", EventType.REGISTRY)
        assert Events("proxy_web", EventType.PROXY_WEB)

    event_paths = [
        (pc_events_json_path, 20),
        (pc_events_jsonl_path, 20),
        (powershell_events_jsonl_path, 30),
    ]

    @pytest.mark.parametrize("events_path,num_events", event_paths)
    def test_load_from_dir(self, events_path, num_events):
        events = Events(EventType.PROCESS_CREATION)
        events.load_from_dir(events_path)
        assert events.size == num_events

    def test_load_from_dir_json(self):
        expected_events_size = 20
        events = Events(EventType.PROCESS_CREATION)
        events.load_from_dir(pc_events_json_path())

        assert events.size == expected_events_size

    def test_load_from_dir_invalid_events_path(self):
        events = Events(EventType.PROCESS_CREATION)

        with pytest.raises(FileNotFoundError):
            events.load_from_dir("some/sample/path")

    def test_load_from_dir_event_type_mismatch(self):
        events = Events(EventType.PROCESS_CREATION)
        events.load_from_dir(powershell_events_jsonl_path())

        assert not events.data

    def test_add_event(self):
        event = {"winlog": {"event_id": 1}}
        events = Events(EventType.PROCESS_CREATION)
        events.add_event(event)

        assert events.size == 1

    split_sizes = [2, [0.5], [0.5, "value"]]

    @pytest.mark.parametrize("split_sizes", split_sizes)
    def test_create_random_split_raising_typerrror(self, split_sizes):
        events = Events(EventType.PROCESS_CREATION)
        events.load_from_dir(pc_events_jsonl_path)

        with pytest.raises(TypeError):
            _ = events.create_random_split(split_sizes=split_sizes)

    split_sizes = [[0.5, -0.4], [0.5, 0.4, 0.3]]

    @pytest.mark.parametrize("split_sizes", split_sizes)
    def test_create_random_split_raising_valuerrror(self, split_sizes):
        events = Events(EventType.PROCESS_CREATION)
        events.load_from_dir(pc_events_jsonl_path())

        with pytest.raises(ValueError):
            _ = events.create_random_split(split_sizes=split_sizes)

    def test_create_random_split_missing_events(self):
        events = Events(EventType.PROCESS_CREATION)
        assert not events.create_random_split(split_sizes=[0.5, 0.5])

    def test_create_random_split_even_number_of_splits(self):
        expected_split_size = 10
        expected_num_splits = 2

        events = Events(EventType.PROCESS_CREATION)
        events.load_from_dir(pc_events_jsonl_path())

        splits = events.create_random_split(split_sizes=[0.5, 0.5])

        assert len(splits) == expected_num_splits
        assert splits[0].size == expected_split_size
        assert splits[1].size == expected_split_size

    def test_create_random_split_uneven_number_of_splits(self):
        expected_num_splits = 3

        events = Events(EventType.PROCESS_CREATION)
        events.load_from_dir(pc_events_jsonl_path())

        splits = events.create_random_split(split_sizes=[0.33, 0.33, 0.33])

        assert len(splits) == expected_num_splits
        assert splits[0].size == 7
        assert splits[1].size == 7
        assert splits[2].size == 6

    def test_create_random_split_with_seed(self):
        events = Events(EventType.PROCESS_CREATION)
        events.load_from_dir(pc_events_jsonl_path())
        splits = events.create_random_split(split_sizes=[0.5, 0.5], seed=42)
        other_splits = events.create_random_split(split_sizes=[0.5, 0.5], seed=42)

        assert len(splits) == len(other_splits)
        assert splits[0].data == other_splits[0].data
        assert splits[1].data == other_splits[1].data


class TestEventsCache:
    def test_add_events(self):
        pc_events = Events(EventType.PROCESS_CREATION, name="process_creation")
        powershell_events = Events(EventType.POWERSHELL, name="powershell")
        registry_events = Events(EventType.REGISTRY, name="registry")
        proxy_events = Events(EventType.PROXY_WEB, name="proxy_web")

        events_cache = EventsCache()

        events_cache.add_events(pc_events)
        events_cache.add_events(powershell_events)
        events_cache.add_events(registry_events)
        events_cache.add_events(proxy_events)

        assert "test" in events_cache.events

    def test_add_events_existing_name(self):
        events_1 = Events(EventType.PROCESS_CREATION, name="test")
        events_2 = Events(EventType.PROCESS_CREATION, name="test")
        events_cache = EventsCache()

        events_cache.add_events(events_1)
        with pytest.raises(EventsError):
            events_cache.add_events(events_2)

    def test_get_events_by_name(self):
        events = Events(EventType.PROCESS_CREATION, name="test")
        events_cache = EventsCache()
        events_cache.add_events(events)

        result = events_cache.get_events_by_name("test")
        assert result.name == "test"

    def test_get_events_by_type(self):
        events = Events(EventType.PROCESS_CREATION, name="test")
        events_cache = EventsCache()
        events_cache.add_events(events)

        result = events_cache.get_events_by_type(EventType.PROCESS_CREATION)
        assert result[0].type == EventType.PROCESS_CREATION

    def test_get_events_by_type_missing_type(self):
        events_cache = EventsCache()
        result = events_cache.get_events_by_type(EventType.PROCESS_CREATION)

        assert not result
