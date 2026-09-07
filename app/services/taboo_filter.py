from app.models.schemas import NameRecommendation


class TabooFilter:
    """Stub for a future cultural taboo and safety filter."""

    def filter(
        self,
        candidates: list[NameRecommendation],
        target_cultures: list[str],
    ) -> tuple[list[NameRecommendation], list[str]]:
        del target_cultures
        return candidates, ["Phase 1 mock data: cultural taboo checks are not yet active."]
