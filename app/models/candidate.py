from pydantic import BaseModel, ConfigDict, Field


class NameCandidate(BaseModel):
    """Internal candidate and retrieval-score model."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    name: str
    culture: str
    gender: str
    ipa: str | None = None
    origin: str | None = None
    etymology: str | None = None
    meaning: str | None = None
    style_tags: list[str] = Field(default_factory=list)
    popularity: float = Field(ge=0, le=1)
    dense_rank: int | None = Field(default=None, ge=1)
    bm25_rank: int | None = Field(default=None, ge=1)
    rrf_score: float = Field(default=0, ge=0)
    normalized_rrf_score: float = Field(default=0, ge=0, le=1)
    phonetic_similarity_score: float = Field(default=0, ge=0, le=1)
    personality_tag_match_score: float = Field(default=0, ge=0, le=1)
    final_score: float = Field(default=0, ge=0, le=1)
