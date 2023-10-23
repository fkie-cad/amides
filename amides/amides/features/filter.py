"""This module contains token elimination classes used to eliminate tokens showing specific
patterns from a list of tokens.
"""

import re
from abc import ABC, abstractmethod


class TokenEliminator(ABC):
    """Base class for all token elimination classes."""

    @abstractmethod
    def __call__(self, token_list):
        pass

    @property
    @abstractmethod
    def name(self):
        """Return the name of the eliminator."""


class NumericValues(TokenEliminator):
    """NumericValues eliminates hex and decimal values whose number of characters/digits exceeds
    a maximum length value."""

    def __init__(self, length):
        """Create instances.

        Parameter
        --------
        length: int
            Maximum character length of hex/decimal values.
        """
        super().__init__()
        self._re = r"^(?:0x)?[0-9a-f]{{{0},}}$".format(length + 1)

    def __call__(self, token_list):
        tokens = [token for token in token_list if not re.match(self._re, token)]

        return tokens

    @property
    def name(self):
        return "hex_values"


class Strings(TokenEliminator):
    """Eliminates strings that exceed a certain length."""

    def __init__(self, length):
        super().__init__()
        self._length = length

    def __call__(self, token_list):
        tokens = [token for token in token_list if not len(token) > self._length]

        return tokens

    @property
    def name(self):
        return "string"
