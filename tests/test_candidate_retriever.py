import csv
from pathlib import Path

import pytest

from app.config.constants import (
    PERSONALITY_TAG_MATCH_WEIGHT,
    PHONETIC_SIMILARITY_WEIGHT,
    POPULARITY_WEIGHT,
)
from app.services.candidate_retriever import CandidateRetriever

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "names.csv"


@pytest.fixture(scope="module")
def retriever() -> CandidateRetriever:
    return CandidateRetriever(DATA_PATH)


def test_knowledge_base_has_required_coverage_and_valid_genders() -> None:
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) >= 50
    assert {row["culture"] for row in rows} == {"en-UK", "en-US", "fr", "ja"}
    assert {row["gender"] for row in rows} <= {"male", "female", "neutral"}


def test_uk_search_returns_only_uk_candidates(
    retriever: CandidateRetriever,
) -> None:
    results = retriever.search_candidates("林", "en-UK", None, [], [], 5)

    assert results
    assert {candidate.culture for candidate in results} == {"en-UK"}


def test_french_search_returns_french_candidates(
    retriever: CandidateRetriever,
) -> None:
    results = retriever.search_candidates("安娜", "fr", "female", [], [], 5)

    assert results
    assert {candidate.culture for candidate in results} == {"fr"}


def test_preferred_initial_is_prioritized(retriever: CandidateRetriever) -> None:
    results = retriever.search_candidates("雨娜", "ja", "female", [], ["Y"], 3)

    assert results[0].name.startswith("Y")


def test_missing_preferred_initial_falls_back(retriever: CandidateRetriever) -> None:
    results = retriever.search_candidates("雨娜", "ja", "female", [], ["X"], 3)

    assert len(results) == 3


def test_yuna_has_high_phonetic_score(retriever: CandidateRetriever) -> None:
    results = retriever.search_candidates("雨娜", "ja", "female", [], ["Y"], 5)
    yuna = next(candidate for candidate in results if candidate.name == "Yuna")

    assert yuna.phonetic_similarity_score > 0.9


def test_elegant_tag_contributes_to_score(retriever: CandidateRetriever) -> None:
    results = retriever.search_candidates(
        "露易丝", "fr", "female", ["elegant"], [], 10
    )
    louise = next(candidate for candidate in results if candidate.name == "Louise")

    assert "elegant" in louise.style_tags
    assert louise.personality_tag_match_score == 1.0
    expected = (
        PHONETIC_SIMILARITY_WEIGHT * louise.phonetic_similarity_score
        + PERSONALITY_TAG_MATCH_WEIGHT
        + POPULARITY_WEIGHT * louise.popularity
    )
    assert louise.final_score == pytest.approx(expected)


def test_top_k_limits_result_count(retriever: CandidateRetriever) -> None:
    results = retriever.search_candidates("李明", "en-US", None, [], [], 3)

    assert len(results) == 3


def test_gender_filter_also_allows_neutral_candidates(
    retriever: CandidateRetriever,
) -> None:
    results = retriever.search_candidates("若文", "en-UK", "female", [], [], 15)

    assert results
    assert {candidate.gender for candidate in results} <= {"female", "neutral"}
