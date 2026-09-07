from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config.constants import DEFAULT_MAX_RECOMMENDATIONS, MAX_RECOMMENDATIONS


class NamingConstraints(BaseModel):
    """Public request constraints with temporary top_k compatibility."""

    model_config = ConfigDict(extra="allow")

    max_recommendations: int = Field(
        default=DEFAULT_MAX_RECOMMENDATIONS,
        ge=1,
        le=MAX_RECOMMENDATIONS,
    )
    preferred_letters: list[str] = Field(default_factory=list)
    desired_meaning: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=MAX_RECOMMENDATIONS)

    def retrieval_limit(self) -> int:
        fields_set = self.model_fields_set
        if "max_recommendations" in fields_set:
            return self.max_recommendations
        if self.top_k is not None:
            return self.top_k
        return self.max_recommendations


class NamingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chinese_name: str = Field(min_length=1, examples=["李明"])
    gender: str = Field(min_length=1, examples=["male"])
    target_cultures: list[str] = Field(min_length=1, examples=[["en-US", "en-UK"]])
    personality_tags: list[str] = Field(default_factory=list, examples=[["calm", "creative"]])
    constraints: NamingConstraints = Field(default_factory=NamingConstraints)


class NameRecommendation(BaseModel):
    name: str
    pronunciation: str
    meaning: str
    cultural_fit: dict[str, str]
    phonetic_similarity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    rationale: str


class NamingResponse(BaseModel):
    chinese_name: str
    recommendations: list[NameRecommendation]
    warnings: list[str]
    latency_ms: float = Field(ge=0)
