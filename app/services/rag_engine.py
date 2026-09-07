import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config.constants import (
    CULTURE_DESCRIPTIONS,
    EMBEDDING_MODEL_NAME,
    HYBRID_PHONETIC_WEIGHT,
    HYBRID_PERSONALITY_WEIGHT,
    HYBRID_POPULARITY_WEIGHT,
    HYBRID_RETRIEVAL_TOP_N,
    HYBRID_RRF_WEIGHT,
    RRF_K,
)
from app.models.candidate import NameCandidate
from app.services.candidate_retriever import CandidateRetriever
from app.services.phonetic import PhoneticService


class BM25Result(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    bm25_score: float
    bm25_rank: int = Field(ge=1)


class DenseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    dense_score: float
    dense_rank: int = Field(ge=1)


class FusedResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    dense_rank: int | None = Field(default=None, ge=1)
    bm25_rank: int | None = Field(default=None, ge=1)
    rrf_score: float = Field(ge=0)


def build_name_document(candidate: NameCandidate) -> str:
    """Create one searchable document from all knowledge-base text fields."""
    return " ".join(
        part
        for part in [
            f"name {candidate.name}",
            f"culture {candidate.culture}",
            f"gender {candidate.gender}",
            f"origin {candidate.origin or ''}",
            f"etymology {candidate.etymology or ''}",
            f"meaning {candidate.meaning or ''}",
            f"style {' '.join(candidate.style_tags)}",
            f"initial_{CandidateRetriever.normalize_initial(candidate.name)}",
        ]
        if part
    )


def build_sparse_query(
    culture: str,
    gender: str | None,
    personality_tags: Sequence[str],
    preferred_letters: Sequence[str],
    desired_meaning: str | None = None,
) -> str:
    terms = [culture, gender or "", *personality_tags, desired_meaning or ""]
    terms.extend(
        f"initial_{initial}"
        for letter in preferred_letters
        if (initial := CandidateRetriever.normalize_initial(letter))
    )
    return " ".join(term for term in terms if term)


def build_dense_query(
    culture: str,
    gender: str | None,
    personality_tags: Sequence[str],
    desired_meaning: str | None = None,
) -> str:
    culture_name = CULTURE_DESCRIPTIONS.get(culture, culture)
    gender_text = f"{gender} " if gender else ""
    style_text = ", ".join(personality_tags) or "open style"
    meaning_text = desired_meaning or "no specific meaning"
    return (
        f"A {gender_text}{culture_name} given name. "
        f"Desired personality and style: {style_text}. "
        f"Desired meaning: {meaning_text}."
    )


class BM25Retriever:
    """Sparse lexical retrieval over pre-tokenized local name documents."""

    def __init__(self, candidates: Sequence[NameCandidate]) -> None:
        self._candidates = list(candidates)
        self._documents = [build_name_document(item) for item in self._candidates]
        self._tokenized_documents = [self.tokenize(text) for text in self._documents]
        self._index = BM25Okapi(self._tokenized_documents)

    def search(
        self,
        query: str,
        candidate_ids: Iterable[str] | None = None,
        top_n: int = HYBRID_RETRIEVAL_TOP_N,
    ) -> list[BM25Result]:
        allowed_ids = set(candidate_ids) if candidate_ids is not None else None
        scores = self._index.get_scores(self.tokenize(query))
        ranked = sorted(
            (
                (index, float(score))
                for index, score in enumerate(scores)
                if (
                    (allowed_ids is None or self._candidates[index].candidate_id in allowed_ids)
                    and score > 0
                )
            ),
            key=lambda item: (-item[1], self._candidates[item[0]].name.casefold()),
        )[:top_n]
        return [
            BM25Result(
                candidate_id=self._candidates[index].candidate_id,
                bm25_score=score,
                bm25_rank=rank,
            )
            for rank, (index, score) in enumerate(ranked, start=1)
        ]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        decomposed = unicodedata.normalize("NFKD", text.casefold())
        plain = "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        )
        return re.findall(r"[a-z0-9_]+", plain)


class DenseRetriever:
    """In-memory multilingual embedding retrieval using cosine similarity."""

    def __init__(
        self,
        candidates: Sequence[NameCandidate],
        model_name: str = EMBEDDING_MODEL_NAME,
        model: Any | None = None,
    ) -> None:
        self._candidates = list(candidates)
        self._model = model or SentenceTransformer(model_name)
        self._documents = [build_name_document(item) for item in self._candidates]
        self._embeddings = np.asarray(
            self._model.encode(
                self._documents,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

    def search(
        self,
        query: str,
        candidate_ids: Iterable[str] | None = None,
        top_n: int = HYBRID_RETRIEVAL_TOP_N,
    ) -> list[DenseResult]:
        allowed_ids = set(candidate_ids) if candidate_ids is not None else None
        query_embedding = np.asarray(
            self._model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        ranked = sorted(
            (
                (index, float(np.dot(embedding, query_embedding)))
                for index, embedding in enumerate(self._embeddings)
                if allowed_ids is None
                or self._candidates[index].candidate_id in allowed_ids
            ),
            key=lambda item: (-item[1], self._candidates[item[0]].name.casefold()),
        )[:top_n]
        return [
            DenseResult(
                candidate_id=self._candidates[index].candidate_id,
                dense_score=score,
                dense_rank=rank,
            )
            for rank, (index, score) in enumerate(ranked, start=1)
        ]


def reciprocal_rank_fusion(
    dense_results: Sequence[DenseResult],
    bm25_results: Sequence[BM25Result],
    k: int = RRF_K,
) -> list[FusedResult]:
    if k < 0:
        raise ValueError("RRF k must be non-negative")

    scores: defaultdict[str, float] = defaultdict(float)
    dense_ranks = {result.candidate_id: result.dense_rank for result in dense_results}
    bm25_ranks = {result.candidate_id: result.bm25_rank for result in bm25_results}

    for result in dense_results:
        scores[result.candidate_id] += 1 / (k + result.dense_rank)
    for result in bm25_results:
        scores[result.candidate_id] += 1 / (k + result.bm25_rank)

    return sorted(
        [
            FusedResult(
                candidate_id=candidate_id,
                dense_rank=dense_ranks.get(candidate_id),
                bm25_rank=bm25_ranks.get(candidate_id),
                rrf_score=score,
            )
            for candidate_id, score in scores.items()
        ],
        key=lambda result: (-result.rrf_score, result.candidate_id),
    )


class HybridRetriever:
    """Hard filters, dense+BM25 RRF, then phonetic-aware final ranking."""

    def __init__(
        self,
        baseline_retriever: CandidateRetriever | None = None,
        phonetic_service: PhoneticService | None = None,
        dense_model: Any | None = None,
    ) -> None:
        self._phonetic_service = phonetic_service or PhoneticService()
        self._baseline = baseline_retriever or CandidateRetriever(
            phonetic_service=self._phonetic_service
        )
        self._candidate_by_id = {
            candidate.candidate_id: candidate for candidate in self._baseline.candidates
        }
        self.bm25 = BM25Retriever(self._baseline.candidates)
        self.dense = DenseRetriever(self._baseline.candidates, model=dense_model)

    def search_candidates(
        self,
        chinese_name: str,
        culture: str,
        gender: str | None,
        personality_tags: list[str],
        preferred_letters: list[str],
        top_k: int,
        desired_meaning: str | None = None,
    ) -> list[NameCandidate]:
        if top_k <= 0:
            return []

        filtered = self._baseline.filter_candidates(culture, gender)
        if not filtered:
            return []
        culture_key = self._baseline.normalize_culture(culture)
        candidate_ids = {candidate.candidate_id for candidate in filtered}
        retrieval_n = min(
            len(filtered),
            max(HYBRID_RETRIEVAL_TOP_N, top_k * 4),
        )
        sparse_query = build_sparse_query(
            culture_key,
            gender,
            personality_tags,
            preferred_letters,
            desired_meaning,
        )
        dense_query = build_dense_query(
            culture_key,
            gender,
            personality_tags,
            desired_meaning,
        )
        bm25_results = self.bm25.search(sparse_query, candidate_ids, retrieval_n)
        dense_results = self.dense.search(dense_query, candidate_ids, retrieval_n)
        fused = reciprocal_rank_fusion(dense_results, bm25_results)
        if not fused:
            return []

        maximum_rrf = max(result.rrf_score for result in fused)
        ranked_candidates = [
            self._finalize_candidate(
                self._candidate_by_id[result.candidate_id],
                result,
                maximum_rrf,
                chinese_name,
                personality_tags,
            )
            for result in fused
        ]
        ranked_candidates.sort(
            key=lambda candidate: (
                -candidate.final_score,
                -candidate.popularity,
                candidate.name.casefold(),
            )
        )
        return self._baseline.prioritize_initials(
            ranked_candidates, preferred_letters
        )[:top_k]

    def _finalize_candidate(
        self,
        candidate: NameCandidate,
        fused: FusedResult,
        maximum_rrf: float,
        chinese_name: str,
        personality_tags: list[str],
    ) -> NameCandidate:
        normalized_rrf = fused.rrf_score / maximum_rrf if maximum_rrf else 0.0
        phonetic_score = self._phonetic_service.calculate_phonetic_similarity(
            chinese_name, candidate.name
        )
        personality_score = self._baseline.personality_match_score(
            personality_tags, candidate.style_tags
        )
        final_score = (
            HYBRID_RRF_WEIGHT * normalized_rrf
            + HYBRID_PHONETIC_WEIGHT * phonetic_score
            + HYBRID_PERSONALITY_WEIGHT * personality_score
            + HYBRID_POPULARITY_WEIGHT * candidate.popularity
        )
        return candidate.model_copy(
            update={
                "dense_rank": fused.dense_rank,
                "bm25_rank": fused.bm25_rank,
                "rrf_score": fused.rrf_score,
                "normalized_rrf_score": normalized_rrf,
                "phonetic_similarity_score": phonetic_score,
                "personality_tag_match_score": personality_score,
                "final_score": final_score,
            }
        )


class RAGEngine(HybridRetriever):
    """Backward-compatible service name for the phase-4 retrieval engine."""
