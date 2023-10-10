from typing import List

from amides.features.preprocessing import FilterDummyCharacters, Lowercase
from amides.features.tokenization import AnyWordCharacter
from amides.features.filter import NumericValues, Strings


def normalize(samples: list[str]) -> list[str]:
    """Normalize list of given samples using the default Normalizer.

    Parameters
    ----------
    samples: list[str]
        List of samples to be normalized


    Returns
    -------
    normalized_samples: list[str]
        List of normalized samples
    """

    normalizer = Normalizer()
    normalized_samples = []

    for sample in samples:
        normalized = normalizer.normalize(sample)
        if normalized:
            normalized_samples.append(normalized)

    return normalized_samples


class Normalizer:
    """Normalizer to convert sample into list of tokens."""

    __slots__ = ("_filter_dummy", "_lower", "_any_word", "_num_values", "_strings")

    def __init__(self, max_len_num_values=3, max_len_strings=30):
        self._filter_dummy = FilterDummyCharacters()
        self._lower = Lowercase()
        self._any_word = AnyWordCharacter()
        self._num_values = NumericValues(length=max_len_num_values)
        self._strings = Strings(length=max_len_strings)

    def normalize(self, sample: str) -> str:
        preprocessed = self._preprocess(sample)
        tokens = self._tokenize(preprocessed)
        shrinked_tokens = self._eliminate_tokens(tokens)

        shrinked_tokens.sort()
        tokens_csv = ",".join(shrinked_tokens)

        return tokens_csv

    def _preprocess(self, sample: str) -> str:
        return self._lower(self._filter_dummy(sample))

    def _tokenize(self, preprocessed: str) -> str:
        return self._any_word(preprocessed)

    def _eliminate_tokens(self, tokens: List[str]) -> List[str]:
        return self._strings(self._num_values(tokens))
