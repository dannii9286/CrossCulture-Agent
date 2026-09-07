from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_generate_names_returns_hybrid_recommendations() -> None:
    response = client.post(
        "/api/v1/generate-names",
        json={
            "chinese_name": "李明",
            "gender": "male",
            "target_cultures": ["US", "UK"],
            "personality_tags": ["calm", "creative"],
            "constraints": {"max_length": 8, "top_k": 3},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["chinese_name"] == "李明"
    assert len(body["recommendations"]) == 6
    assert all(
        set(recommendation["cultural_fit"]) <= {"en-US", "en-UK"}
        for recommendation in body["recommendations"]
    )
    assert body["latency_ms"] >= 0
    assert body["warnings"]


def test_generate_names_rejects_missing_required_fields() -> None:
    response = client.post(
        "/api/v1/generate-names",
        json={"chinese_name": "李明"},
    )

    assert response.status_code == 422


def test_api_preserves_preferred_initial_priority() -> None:
    response = client.post(
        "/api/v1/generate-names",
        json={
            "chinese_name": "雨娜",
            "gender": "female",
            "target_cultures": ["ja"],
            "personality_tags": [],
            "constraints": {
                "preferred_letters": ["Y"],
                "max_recommendations": 3,
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["recommendations"][0]["name"] == "Yuna"
