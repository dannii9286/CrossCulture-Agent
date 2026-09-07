import re
import unicodedata
from typing import Protocol, TypedDict

from pypinyin import Style, lazy_pinyin


class PhoneticAnalysis(TypedDict):
    """Stable phase-2 result contract for a Chinese-name analysis."""

    original_name: str
    pinyin: str
    normalized_phonetic: str
    ipa: str | None


class PhoneticNormalizer(Protocol):
    """Replacement point for a future IPA/G2P-backed normalizer."""

    def normalize(self, text: str) -> str:
        """Convert text to a representation suitable for distance comparison."""


class BasicPhoneticNormalizer:
    """Simple, deterministic Latin phonetic normalization.

    This is intentionally not IPA. It removes tone/diacritic marks, punctuation,
    whitespace, and case differences. Pinyin ``ü``, ``u:``, and ``v`` spellings
    are folded to ``u`` so common keyboard variants compare consistently.
    """

    _NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]")

    def normalize(self, text: str) -> str:
        lowered = text.casefold().replace("u:", "u").replace("v", "u")
        decomposed = unicodedata.normalize("NFKD", lowered)
        without_marks = "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        )
        return self._NON_ALPHANUMERIC.sub("", without_marks)


def levenshtein_distance(left: str, right: str) -> int:
    """Return the Levenshtein edit distance using O(min(n, m)) memory."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    if len(left) < len(right):
        left, right = right, left

    previous_row = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            insertion = current_row[right_index - 1] + 1
            deletion = previous_row[right_index] + 1
            substitution = (
                previous_row[right_index - 1]
                + (left_character != right_character)
            )
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row

    return previous_row[-1]


class PhoneticService:
    """Analyze Chinese names and compare their pronunciation spellings."""

    def __init__(self, normalizer: PhoneticNormalizer | None = None) -> None:
        self._normalizer = normalizer or BasicPhoneticNormalizer()

    def analyze(self, chinese_name: str) -> PhoneticAnalysis:
        syllables = lazy_pinyin(
            chinese_name,
            style=Style.NORMAL,
            strict=True,
            errors=lambda characters: list(characters),
        )
        standard_pinyin = " ".join(syllables)
        return {
            "original_name": chinese_name,
            "pinyin": standard_pinyin,
            "normalized_phonetic": self._normalizer.normalize(standard_pinyin),
            # Real multilingual IPA needs a dedicated, validated G2P backend.
            "ipa": None,
        }

    def normalize(self, text: str) -> str:
        """Expose the active normalization strategy for independent testing."""
        return self._normalizer.normalize(text)

    def calculate_phonetic_similarity(
        self,
        chinese_name: str,
        candidate_name: str,
    ) -> float:
        source = self.analyze(chinese_name)["normalized_phonetic"]
        candidate = self._normalizer.normalize(candidate_name)
        longest_length = max(len(source), len(candidate))

        if longest_length == 0:
            return 1.0

        distance = levenshtein_distance(source, candidate)
        return 1.0 - (distance / longest_length)


def calculate_phonetic_similarity(
    chinese_name: str,
    candidate_name: str,
) -> float:
    """Convenience API using the default phonetic normalization strategy."""
    return PhoneticService().calculate_phonetic_similarity(
        chinese_name,
        candidate_name,
    )
