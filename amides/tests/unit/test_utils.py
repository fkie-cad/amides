import pytest
from amides.utils import TimeRangeIterator


class TestTimeRangeIterator:
    iterate = [
        (
            "2022-05-31T00:00:00",
            "2022-06-01T00:00:00",
            "12:00:00",
            [
                ("2022-05-31T00:00:00", "2022-05-31T12:00:00"),
                ("2022-05-31T12:00:00", "2022-06-01T00:00:00"),
            ],
        ),
        (
            "2023-06-01T00:00:00",
            "2023-06-01T00:00:05",
            "00:00:01",
            [
                ("2023-06-01T00:00:00", "2023-06-01T00:00:01"),
                ("2023-06-01T00:00:01", "2023-06-01T00:00:02"),
                ("2023-06-01T00:00:02", "2023-06-01T00:00:03"),
                ("2023-06-01T00:00:03", "2023-06-01T00:00:04"),
                ("2023-06-01T00:00:04", "2023-06-01T00:00:05"),
            ],
        ),
    ]

    @pytest.mark.parametrize("start,end,interval,expected", iterate)
    def test_create_time_ranges(self, start, end, interval, expected):
        time_ranger = TimeRangeIterator(start=start, end=end, interval=interval)
        time_ranges = [(start, end) for start, end in time_ranger.next()]

        assert time_ranges == expected
