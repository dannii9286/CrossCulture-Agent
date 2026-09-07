import csv
import unicodedata
from pathlib import Path
from typing import Iterable

from app.config.constants import (
    CULTURE_ALIASES,
    PERSONALITY_TAG_MATCH_WEIGHT,
    PHONETIC_SIMILARITY_WEIGHT,
    POPULARITY_WEIGHT,
)
from app.models.candidate import NameCandidate
from app.services.phonetic import PhoneticService


class CandidateRetriever:
    """Explainable phase-3 baseline retained alongside hybrid retrieval."""

    def __init__(
        self,
        data_path: Path | str | None = None,
        phonetic_service: PhoneticService | None = None,
    ) -> None:
        default_path = Path(__file__).resolve().parents[2] / "data" / "names.csv"
        self._data_path = Path(data_path) if data_path else default_path
        self._phonetic_service = phonetic_service or PhoneticService()
        self._candidates = self._load_candidates()

    @property
    def candidates(self) -> tuple[NameCandidate, ...]:
        return tuple(self._candidates)

    def search_candidates(
        self,
        chinese_name: str,
        culture: str,
        gender: str | None,
        personality_tags: list[str],
        preferred_letters: list[str],
        top_k: int,
    ) -> list[NameCandidate]:
        if top_k <= 0:
            return []

        culture_matches = self.filter_candidates(culture, gender)
        scored = [
            self._score_candidate(candidate, chinese_name, personality_tags)
            for candidate in culture_matches
        ]
        scored.sort(key=self._sort_key)
        return self.prioritize_initials(scored, preferred_letters)[:top_k]

    def filter_candidates(
        self,
        culture: str,
        gender: str | None,
    ) -> list[NameCandidate]:
        normalized_culture = self.normalize_culture(culture)
        matches = [
            candidate
            for candidate in self._candidates
            if candidate.culture == normalized_culture
        ]
        normalized_gender = gender.casefold() if gender else None
        if normalized_gender:
            matches = [
                candidate
                for candidate in matches
                if candidate.gender in {normalized_gender, "neutral"}
            ]
        return matches

    @classmethod
    def prioritize_initials(
        cls,
        candidates: list[NameCandidate],
        preferred_letters: list[str],
    ) -> list[NameCandidate]:
        preferred = {
            initial
            for letter in preferred_letters
            if (initial := cls.normalize_initial(letter))
        }
        if not preferred:
            return candidates
        initial_matches = [
            candidate
            for candidate in candidates
            if cls.normalize_initial(candidate.name) in preferred
        ]
        fallback_matches = [
            candidate
            for candidate in candidates
            if cls.normalize_initial(candidate.name) not in preferred
        ]
        return initial_matches + fallback_matches

    def _load_candidates(self) -> list[NameCandidate]:
        if not self._data_path.is_file():
            raise FileNotFoundError(f"Name knowledge base not found: {self._data_path}")

        with self._data_path.open(encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))

        popularities = self._normalize_popularities(
            [float(row["popularity"]) for row in rows]
        )
        candidates: list[NameCandidate] = []
        for index, row in enumerate(rows):
            name = row["name"].strip()
            culture = row["culture"].strip()
            gender = row["gender"].strip().casefold()
            candidates.append(
                NameCandidate(
                    candidate_id=f"{culture}:{gender}:{name.casefold()}",
                    name=name,
                    culture=culture,
                    gender=gender,
                    ipa=self._optional(row["ipa"]),
                    origin=self._optional(row["origin"]),
                    etymology=self._optional(row["etymology"]),
                    meaning=self._optional(row["meaning"]),
                    style_tags=self._parse_tags(row["style_tags"]),
                    popularity=popularities[index],
                )
            )
        return candidates

    def _score_candidate(
        self,
        candidate: NameCandidate,
        chinese_name: str,
        personality_tags: list[str],
    ) -> NameCandidate:
        phonetic_score = self._phonetic_service.calculate_phonetic_similarity(
            chinese_name, candidate.name
        )
        personality_score = self.personality_match_score(
            personality_tags, candidate.style_tags
        )
        final_score = (
            PHONETIC_SIMILARITY_WEIGHT * phonetic_score
            + PERSONALITY_TAG_MATCH_WEIGHT * personality_score
            + POPULARITY_WEIGHT * candidate.popularity
        )
        return candidate.model_copy(
            update={
                "phonetic_similarity_score": phonetic_score,
                "personality_tag_match_score": personality_score,
                "final_score": final_score,
            }
        )

    @staticmethod
    def personality_match_score(
        requested_tags: Iterable[str],
        candidate_tags: Iterable[str],
    ) -> float:
        requested = {tag.strip().casefold() for tag in requested_tags if tag.strip()}
        if not requested:
            return 0.0
        available = {tag.strip().casefold() for tag in candidate_tags if tag.strip()}
        return len(requested & available) / len(requested)

    @staticmethod
    def _normalize_popularities(values: list[float]) -> list[float]:
        if not values:
            return []
        if all(0 <= value <= 1 for value in values):
            return values
        minimum = min(values)
        maximum = max(values)
        if minimum == maximum:
            return [1.0 for _ in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    @staticmethod
    def normalize_culture(culture: str) -> str:
        stripped = culture.strip()
        return CULTURE_ALIASES.get(stripped.casefold(), stripped)

    @staticmethod
    def normalize_initial(value: str) -> str:
        decomposed = unicodedata.normalize("NFKD", value.strip().casefold())
        letters = "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character) and character.isalpha()
        )
        return letters[:1]

    @staticmethod
    def _parse_tags(value: str) -> list[str]:
        return [tag.strip().casefold() for tag in value.split("|") if tag.strip()]

    @staticmethod
    def _optional(value: str) -> str | None:
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _sort_key(candidate: NameCandidate) -> tuple[float, float, str]:
        return (-candidate.final_score, -candidate.popularity, candidate.name.casefold())
