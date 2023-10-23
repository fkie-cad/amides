"""This module contains classes used to turn samples into lists of tokens."""
import re
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Base class for all tokenization-classes."""

    @abstractmethod
    def __call__(self, string):
        pass

    @property
    @abstractmethod
    def name(self):
        """Returns unique name tag of the tokenizer-class."""


class Split(Tokenizer):
    """Split-Tokenizer to be used as tokenizer by Vectorizer-classes
    to split tokens on whitespace characters."""

    def __call__(self, string):
        return string.split()

    @property
    def name(self):
        return "split"


class WhitespaceAsterisk(Tokenizer):
    """WsAstTokenizer to be used as tokenizer in order to split
    strings on whitespaces and asterisk (*)."""

    def __init__(self):
        super().__init__()
        self._re = r"([^\*\s]+)"

    def __call__(self, string):
        return re.findall(self._re, string)

    @property
    def name(self):
        return "ws_ast"


class WhitespaceAsteriskSlashMinus(Tokenizer):
    """WhitespaceAsteriskSlashMinus-Tokenizer splits strings
    on whitespace, asterisk(*), slashes (/\), and minus (-) symbols.
    """

    def __init__(self):
        super().__init__()
        self._re = r"([^\s\*\\/-]+)"

    def __call__(self, string):
        return re.findall(self._re, string)

    @property
    def name(self):
        return "ws_ast_sla_min"


class WhitespaceAsteriskSlashMinusEquals(Tokenizer):
    """WhitespaceAsteriskSlashMinusEquals-Tokenizer splits strings
    on whitespace, asterisk(*), slashes (/\), minus (-), and equals (=) symbols.
    """

    def __init__(self):
        super().__init__()
        self._re = r"([^\s\\/\*=-]+)"

    def __call__(self, string):
        return re.findall(self._re, string)

    @property
    def name(self):
        return "ws_ast_sla_min_eq"


class AnyWordCharacter(Tokenizer):
    """Split string samples on the occurrence of any-word character,
    i.e.[a-zA-Z_0-9]."""

    def __init__(self):
        super().__init__()
        self._re = r"(\w+)"

    def __call__(self, string):
        return re.findall(self._re, string)

    @property
    def name(self):
        return "any_word_char"


class CommaSeparation(Tokenizer):
    """Split samples on any comma value."""

    def __call__(self, string):
        return string.split(",")

    @property
    def name(self):
        return "comma_separation"


class TokenizerFactory:
    """TokenizerFactory to create Tokenizer-Objects using their unique name tags."""

    _tokenizers = {
        "split": Split,
        "ws_ast": WhitespaceAsterisk,
        "ws_ast_sla_min": WhitespaceAsteriskSlashMinus,
        "ws_ast_sla_min_eq": WhitespaceAsteriskSlashMinusEquals,
        "any_word_char": AnyWordCharacter,
        "comma_separation": CommaSeparation,
    }

    @classmethod
    def create(cls, name):
        """Create a Tokenizer-Instance using its unique name tag.

        Parameters
        ----------
        name: str
            The unique name of the Tokenizer-class.

        Returns
        ------
            : Tokenizer
        """
        return cls._tokenizers[name]()
