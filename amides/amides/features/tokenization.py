import re
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def __call__(self, string):
        pass

    @property
    @abstractmethod
    def name(self):
        pass


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
    """WhitespaceAsteriskSlashMinus-Tokenizer to be used as tokenizer. Splits strings
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
    def __init__(self):
        super().__init__()
        self._re = r"([^\s\\/\*=-]+)"

    def __call__(self, string):
        return re.findall(self._re, string)

    @property
    def name(self):
        return "ws_ast_sla_min_eq"


class AnyWordCharacter(Tokenizer):
    def __init__(self):
        super().__init__()
        self._re = r"(\w+)"

    def __call__(self, string):
        return re.findall(self._re, string)

    @property
    def name(self):
        return "any_word_char"


class CommaSeparation(Tokenizer):
    def __call__(self, string):
        return string.split(",")

    @property
    def name(self):
        return "comma_separation"


class TokenizerFactory:

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
        return cls._tokenizers[name]()
