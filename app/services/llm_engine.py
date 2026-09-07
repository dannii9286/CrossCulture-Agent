from typing import Any

from app.models.schemas import NameRecommendation, NamingRequest


class LLMEngine:
    """Stub for a future LLM-backed naming service."""

    def generate(
        self,
        request: NamingRequest,
        phonetic_profile: dict[str, Any],
        cultural_context: dict[str, Any],
    ) -> list[NameRecommendation]:
        del phonetic_profile, cultural_context
        cultural_fit = {
            culture: "Mock assessment: familiar and easy to pronounce."
            for culture in request.target_cultures
        }
        return [
            NameRecommendation(
                name="Ming",
                pronunciation="ming",
                meaning="Bright and clear (mock explanation).",
                cultural_fit=cultural_fit,
                phonetic_similarity=0.95,
                confidence=0.80,
                rationale=(
                    "Mock recommendation preserving a recognizable sound from the "
                    f"original name {request.chinese_name}."
                ),
            ),
            NameRecommendation(
                name="Milo",
                pronunciation="MY-loh",
                meaning="A friendly international-style option (mock explanation).",
                cultural_fit=cultural_fit,
                phonetic_similarity=0.65,
                confidence=0.72,
                rationale="Mock alternative selected for brevity and easy pronunciation.",
            ),
        ]
