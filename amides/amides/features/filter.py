import re
from abc import ABC, abstractmethod


class TokenEliminator(ABC):
    @abstractmethod
    def __call__(self, token_list):
        pass

    @property
    @abstractmethod
    def name(self):
        pass


class NumericValues(TokenEliminator):
    def __init__(self, length):
        super().__init__()
        self._re = r"^(?:0x)?[0-9a-f]{{{0},}}$".format(length + 1)

    def __call__(self, token_list):
        tokens = [token for token in token_list if not re.match(self._re, token)]

        return tokens

    @property
    def name(self):
        return "hex_values"


class Strings(TokenEliminator):
    def __init__(self, length):
        super().__init__()
        self._length = length

    def __call__(self, token_list):
        tokens = [token for token in token_list if not len(token) > self._length]

        return tokens

    @property
    def name(self):
        return "string"
