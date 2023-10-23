"""This module contains classes used for preprocessing during normalization."""
import re
from abc import ABC, abstractmethod


class Preprocessor(ABC):
    """Base class for all Preprocessing-classes."""

    @abstractmethod
    def __call__(self, string):
        pass

    @property
    @abstractmethod
    def name(self):
        """Return name of the preprocessor."""


class FilterDummyCharacters(Preprocessor):
    """FilterDummyCharacter removes all command-line dummy characters (",^,`)."""

    def __init__(self):
        super().__init__()
        self._re = r"[\"\^`’]"

    def __call__(self, string):
        return re.sub(self._re, "", string)

    @property
    def name(self):
        return "dummy_chars"


class Lowercase(Preprocessor):
    """Turns all samples into lowercase."""

    def __call__(self, string):
        return string.lower()

    @property
    def name(self):
        return "lowercase"
