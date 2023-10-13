import re
from abc import ABC, abstractmethod


class Preprocessor(ABC):
    @abstractmethod
    def __call__(self, string):
        pass

    @property
    @abstractmethod
    def name(self):
        pass


class FilterDummyCharacters(Preprocessor):
    def __init__(self):
        super().__init__()
        self._re = r"[\"\^`’]"

    def __call__(self, string):
        return re.sub(self._re, "", string)

    @property
    def name(self):
        return "dummy_chars"


class Lowercase(Preprocessor):
    def __call__(self, string):
        return string.lower()

    @property
    def name(self):
        return "lowercase"
