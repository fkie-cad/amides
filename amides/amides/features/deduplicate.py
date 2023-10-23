""" This module contains functions and classes that help to deduplicate samples.
"""
import json
from amides.features.normalize import Normalizer


class Cache:
    """Simple cache based on a dict to hold already seen strings and
    count their total number of occurrences.
    """

    __slots__ = ("_elements",)

    def __init__(self):
        self._elements = {}

    @property
    def elements(self) -> dict:
        """Returns elements held by the cache."""
        return self._elements

    @property
    def samples(self) -> list[str]:
        """Returns the unique samples."""
        return list(self._elements.keys())

    def insert(self, element: str):
        """Insert element into the cache. Increases count
        if value has been seen before.

        Parameters
        ----------
        element :str
            String value that should be insterted
        """
        self._elements[element] = self._elements.get(element, 0) + 1

    def get(self, element: str) -> int:
        """Return specific element from the cache in case this element
        has been seen before.

        Parameters
        ----------
        element: str
            String value

        Returns
        -------
            : str
        """
        return self._elements.get(element, "")


def deduplicate_samples(samples_path: str, normalizer: Normalizer) -> Cache:
    """Deduplicates list of samples in the specified file using Cache. Samples
    are normalized first using the provided Normalizer-instance.

    Parameters
    ----------
    samples_path :str
        Path of the file containing samples
    normalizer :Normalizer
        Normalizer instance to normalize samples prior to deduplication

    Returns
    -------
    cache :Cache
        Cache object holding the deduplicated samples
    """
    cache = Cache()

    for sample in open(samples_path, "r", encoding="utf-8"):
        try:
            loaded = json.loads(sample.rstrip("\n"))
        except json.JSONDecodeError:
            continue

        normalized = normalizer.normalize(loaded)
        if not normalized:
            continue

        cache.insert(normalized)

    return cache
