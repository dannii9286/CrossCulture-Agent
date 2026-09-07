# Phase 3 baseline scoring (kept for CandidateRetriever compatibility).
PHONETIC_SIMILARITY_WEIGHT = 0.7
PERSONALITY_TAG_MATCH_WEIGHT = 0.2
POPULARITY_WEIGHT = 0.1

# Phase 4 hybrid final-ranking weights.
HYBRID_RRF_WEIGHT = 0.45
HYBRID_PHONETIC_WEIGHT = 0.35
HYBRID_PERSONALITY_WEIGHT = 0.15
HYBRID_POPULARITY_WEIGHT = 0.05

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
RRF_K = 60
HYBRID_RETRIEVAL_TOP_N = 20

DEFAULT_MAX_RECOMMENDATIONS = 3
MAX_RECOMMENDATIONS = 20

# Older internal names remain available to avoid breaking baseline callers.
DEFAULT_TOP_K = 5
MAX_TOP_K = MAX_RECOMMENDATIONS

CULTURE_ALIASES = {
    "uk": "en-UK",
    "en-uk": "en-UK",
    "us": "en-US",
    "en-us": "en-US",
    "fr": "fr",
    "ja": "ja",
    "jp": "ja",
}

CULTURE_DESCRIPTIONS = {
    "en-UK": "British English",
    "en-US": "American English",
    "fr": "French",
    "ja": "Japanese",
}
