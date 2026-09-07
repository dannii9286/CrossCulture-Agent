from time import perf_counter

from fastapi import APIRouter

from app.models.candidate import NameCandidate
from app.models.schemas import NameRecommendation, NamingRequest, NamingResponse
from app.services.candidate_retriever import CandidateRetriever
from app.services.phonetic import PhoneticService
from app.services.rag_engine import HybridRetriever
from app.services.taboo_filter import TabooFilter

router = APIRouter()

phonetic_service = PhoneticService()
baseline_retriever = CandidateRetriever(phonetic_service=phonetic_service)
hybrid_retriever = HybridRetriever(
    baseline_retriever=baseline_retriever,
    phonetic_service=phonetic_service,
)
taboo_filter = TabooFilter()


@router.post("/generate-names", response_model=NamingResponse)
def generate_names(request: NamingRequest) -> NamingResponse:
    """Run per-culture hybrid retrieval without an LLM."""
    started_at = perf_counter()

    phonetic_service.analyze(request.chinese_name)
    max_recommendations = request.constraints.retrieval_limit()

    selected: list[NameCandidate] = []
    for culture in request.target_cultures:
        selected.extend(
            hybrid_retriever.search_candidates(
                chinese_name=request.chinese_name,
                culture=culture,
                gender=request.gender,
                personality_tags=request.personality_tags,
                preferred_letters=request.constraints.preferred_letters,
                desired_meaning=request.constraints.desired_meaning,
                top_k=max_recommendations,
            )
        )

    recommendations = [_to_recommendation(candidate) for candidate in selected]
    safe_candidates, warnings = taboo_filter.filter(
        candidates=recommendations,
        target_cultures=request.target_cultures,
    )

    latency_ms = round((perf_counter() - started_at) * 1000, 3)
    return NamingResponse(
        chinese_name=request.chinese_name,
        recommendations=safe_candidates,
        warnings=warnings,
        latency_ms=latency_ms,
    )


def _to_recommendation(candidate: NameCandidate) -> NameRecommendation:
    tag_summary = ", ".join(candidate.style_tags) or "none"
    cultural_reason = (
        f"Hybrid retrieval match for {candidate.culture}; style tags: {tag_summary}."
    )
    details = [
        f"Origin: {candidate.origin}." if candidate.origin else "",
        f"Etymology: {candidate.etymology}." if candidate.etymology else "",
        (
            "Hybrid score components: "
            f"dense_rank={candidate.dense_rank}, "
            f"bm25_rank={candidate.bm25_rank}, "
            f"normalized_rrf={candidate.normalized_rrf_score:.3f}, "
            f"phonetic={candidate.phonetic_similarity_score:.3f}, "
            f"personality={candidate.personality_tag_match_score:.3f}, "
            f"popularity={candidate.popularity:.3f}."
        ),
    ]
    return NameRecommendation(
        name=candidate.name,
        pronunciation=candidate.ipa or "",
        meaning=candidate.meaning or "",
        cultural_fit={candidate.culture: cultural_reason},
        phonetic_similarity=candidate.phonetic_similarity_score,
        confidence=candidate.final_score,
        rationale=" ".join(detail for detail in details if detail),
    )
