import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.phonetic import (
    PhoneticService,
    calculate_phonetic_similarity,
    levenshtein_distance,
)

client = TestClient(app)


def test_analyze_chinese_name_returns_pinyin_and_normalized_form() -> None:
    analysis = PhoneticService().analyze("林雨桐")

    assert analysis["original_name"] == "林雨桐"
    assert analysis["pinyin"].split() == ["lin", "yu", "tong"]
    assert analysis["normalized_phonetic"] == "linyutong"
    assert analysis["ipa"] is None


def test_yuna_is_closer_to_yuna_chinese_name_than_clara() -> None:
    yuna_score = calculate_phonetic_similarity("雨娜", "Yuna")
    clara_score = calculate_phonetic_similarity("雨娜", "Clara")

    assert yuna_score > clara_score


@pytest.mark.parametrize(
    ("chinese_name", "candidate_name", "minimum_score"),
    [
        ("雨娜", "Yuna", 0.99),
        ("李明", "Liming", 0.99),
        ("林雨桐", "Lin Yutong", 0.99),
    ],
)
def test_same_or_nearly_identical_pronunciations_score_near_one(
    chinese_name: str,
    candidate_name: str,
    minimum_score: float,
) -> None:
    assert calculate_phonetic_similarity(chinese_name, candidate_name) >= minimum_score


def test_unrelated_name_has_a_clearly_lower_score() -> None:
    related_score = calculate_phonetic_similarity("雨娜", "Yuna")
    unrelated_score = calculate_phonetic_similarity("雨娜", "Christopher")

    assert unrelated_score < 0.5
    assert unrelated_score < related_score


def test_levenshtein_distance_known_example() -> None:
    assert levenshtein_distance("kitten", "sitting") == 3


def test_api_pipeline_uses_calculated_phonetic_score() -> None:
    response = client.post(
        "/api/v1/generate-names",
        json={
            "chinese_name": "李明",
            "gender": "male",
            "target_cultures": ["US"],
            "personality_tags": [],
            "constraints": {},
        },
    )

    assert response.status_code == 200
    recommendation = response.json()["recommendations"][0]
    expected = calculate_phonetic_similarity("李明", recommendation["name"])
    assert recommendation["phonetic_similarity"] == pytest.approx(expected)
