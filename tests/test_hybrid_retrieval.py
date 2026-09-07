import pytest
from fastapi.testclient import TestClient

from app.api.v1.routes import baseline_retriever, hybrid_retriever
from app.config.constants import RRF_K
from app.main import app
from app.services.rag_engine import (
    BM25Result,
    DenseResult,
    build_dense_query,
    build_sparse_query,
    reciprocal_rank_fusion,
)

client = TestClient(app)


def test_bm25_returns_lexical_matching_candidate() -> None:
    allowed = baseline_retriever.filter_candidates("fr", "female")
    query = build_sparse_query("fr", "female", ["elegant"], ["L"])
    results = hybrid_retriever.bm25.search(
        query,
        {candidate.candidate_id for candidate in allowed},
        top_n=5,
    )
    names = {
        hybrid_retriever._candidate_by_id[result.candidate_id].name
        for result in results
    }

    assert "Louise" in names


def test_dense_returns_semantically_related_bright_candidate() -> None:
    allowed = baseline_retriever.filter_candidates("en-US", "female")
    query = build_dense_query(
        "en-US",
        "female",
        ["radiant", "luminous"],
        "light and brightness",
    )
    results = hybrid_retriever.dense.search(
        query,
        {candidate.candidate_id for candidate in allowed},
        top_n=5,
    )
    candidates = [
        hybrid_retriever._candidate_by_id[result.candidate_id]
        for result in results
    ]

    assert any("bright" in candidate.style_tags for candidate in candidates)


def test_rrf_rewards_candidate_present_in_both_rankings() -> None:
    dense = [
        DenseResult(candidate_id="both", dense_score=0.9, dense_rank=1),
        DenseResult(candidate_id="dense-only", dense_score=0.8, dense_rank=2),
    ]
    sparse = [
        BM25Result(candidate_id="both", bm25_score=2.0, bm25_rank=2),
        BM25Result(candidate_id="sparse-only", bm25_score=1.0, bm25_rank=1),
    ]

    fused = reciprocal_rank_fusion(dense, sparse)
    scores = {result.candidate_id: result.rrf_score for result in fused}

    assert scores["both"] == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 2))
    assert scores["both"] > scores["dense-only"]
    assert scores["both"] > scores["sparse-only"]


def test_phonetic_score_participates_in_final_ranking() -> None:
    results = hybrid_retriever.search_candidates(
        "雨娜", "ja", "female", ["bright"], [], 3
    )

    assert results[0].name == "Yuna"
    assert results[0].phonetic_similarity_score > 0.9


def test_hybrid_preserves_culture_filter() -> None:
    results = hybrid_retriever.search_candidates(
        "露易丝", "fr", "female", ["elegant"], [], 5
    )

    assert results
    assert {candidate.culture for candidate in results} == {"fr"}


def test_api_returns_up_to_k_for_each_culture() -> None:
    response = client.post(
        "/api/v1/generate-names",
        json={
            "chinese_name": "安娜",
            "gender": "female",
            "target_cultures": ["en-UK", "fr"],
            "personality_tags": ["elegant"],
            "constraints": {"max_recommendations": 3},
        },
    )

    assert response.status_code == 200
    recommendations = response.json()["recommendations"]
    assert len(recommendations) == 6
    cultures = [next(iter(item["cultural_fit"])) for item in recommendations]
    assert cultures.count("en-UK") == 3
    assert cultures.count("fr") == 3


def test_max_recommendations_is_enforced() -> None:
    response = client.post(
        "/api/v1/generate-names",
        json={
            "chinese_name": "李明",
            "gender": "male",
            "target_cultures": ["en-US"],
            "personality_tags": [],
            "constraints": {"max_recommendations": 2},
        },
    )

    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 2


def test_max_recommendations_defaults_to_three() -> None:
    response = client.post(
        "/api/v1/generate-names",
        json={
            "chinese_name": "李明",
            "gender": "male",
            "target_cultures": ["en-US"],
            "personality_tags": [],
            "constraints": {},
        },
    )

    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 3


def test_hybrid_preferred_letter_falls_back_when_no_match() -> None:
    results = hybrid_retriever.search_candidates(
        "雨娜", "ja", "female", [], ["X"], 3
    )

    assert len(results) == 3
