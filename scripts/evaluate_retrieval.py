import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.candidate_retriever import CandidateRetriever  # noqa: E402
from app.services.rag_engine import HybridRetriever  # noqa: E402


def calculate_metrics(
    ranked_names: list[list[str]],
    relevant_names: list[set[str]],
) -> dict[str, float]:
    query_count = len(ranked_names)
    recall_at_3 = sum(
        bool(set(names[:3]) & relevant)
        for names, relevant in zip(ranked_names, relevant_names)
    ) / query_count
    recall_at_5 = sum(
        bool(set(names[:5]) & relevant)
        for names, relevant in zip(ranked_names, relevant_names)
    ) / query_count
    reciprocal_ranks = []
    for names, relevant in zip(ranked_names, relevant_names):
        first_rank = next(
            (rank for rank, name in enumerate(names, start=1) if name in relevant),
            None,
        )
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
    return {
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "mrr": sum(reciprocal_ranks) / query_count,
    }


def main() -> None:
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "retrieval_cases.json"
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))

    baseline = CandidateRetriever()
    hybrid = HybridRetriever(baseline_retriever=baseline)
    baseline_rankings: list[list[str]] = []
    hybrid_rankings: list[list[str]] = []
    relevant_names: list[set[str]] = []

    for case in cases:
        common = {
            "chinese_name": case["chinese_name"],
            "culture": case["culture"],
            "gender": case["gender"],
            "personality_tags": case["personality_tags"],
            "preferred_letters": [],
            "top_k": 5,
        }
        baseline_results = baseline.search_candidates(**common)
        hybrid_results = hybrid.search_candidates(
            **common,
            desired_meaning=case.get("desired_meaning"),
        )
        baseline_rankings.append([candidate.name for candidate in baseline_results])
        hybrid_rankings.append([candidate.name for candidate in hybrid_results])
        relevant_names.append(set(case["expected_names"]))

    report = {
        "query_count": len(cases),
        "baseline": calculate_metrics(baseline_rankings, relevant_names),
        "hybrid": calculate_metrics(hybrid_rankings, relevant_names),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
