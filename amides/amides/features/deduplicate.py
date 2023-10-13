import json
from amides.features.normalize import Normalizer


def deduplicate_samples(samples_path: str, normalizer: Normalizer) -> "Cache":
    cache = Cache()

    for sample in open(samples_path, "r"):
        try:
            loaded = json.loads(sample.rstrip("\n"))
        except json.JSONDecodeError:
            continue

        normalized = normalizer.normalize(loaded)
        if not normalized:
            continue

        cache.insert(normalized)

    return cache


class Cache:
    """Simple cache to hold already seen samples and count their number of occurrences."""

    __slots__ = ("_elements",)

    def __init__(self):
        self._elements = {}

    @property
    def elements(self) -> dict:
        return self._elements

    @property
    def samples(self) -> list[str]:
        return list(self._elements.keys())

    def insert(self, element: str):
        self._elements[element] = self._elements.get(element, 0) + 1

    def get(self, element: str) -> int:
        return self._elements.get(element, "")
