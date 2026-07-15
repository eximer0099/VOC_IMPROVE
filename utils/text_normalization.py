"""Deterministic Korean typo and colloquial normalization for VOC search."""

from __future__ import annotations

import re


# Longer phrases are applied first. This dictionary is intentionally small and
# auditable; additions should be backed by real VOC wording and regression tests.
COLLOQUIAL_REPLACEMENTS = {
    "주문안보여요": "주문 보이지 않아요",
    "안보여요": "보이지 않아요",
    "안보임": "보이지 않음",
    "안돼요": "되지 않아요",
    "안되요": "되지 않아요",
    "안됨": "되지 않음",
    "안옴": "오지 않음",
    "언제와요": "언제 오나요",
    "먹통": "작동하지 않음",
    "튕겨요": "강제 종료돼요",
    "튕김": "강제 종료",
    "버벅여요": "느려요",
    "겁나": "매우",
    "넘": "너무",
}

SPELLING_REPLACEMENTS = {
    "됫": "됐",
    "됬": "됐",
    "는대": "는데",
    "데요": "대요",
    "왓": "왔",
    "머에요": "뭐예요",
}

SEARCH_STOPWORDS = {
    "그냥", "계속", "자꾸", "너무", "매우", "정말", "이렇게", "저렇게",
    "해주세요", "해줘", "알려주세요", "문의", "관련", "문제", "않아요", "않음",
}


def normalize_korean_text(text: str) -> str:
    """Normalize common typos, spacing, and colloquial VOC expressions."""
    normalized = str(text or "").lower().strip()
    for source, target in sorted(
        COLLOQUIAL_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        normalized = normalized.replace(source, target)
    for source, target in SPELLING_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def build_search_terms(values: list[str]) -> list[str]:
    """Expand input phrases into normalized, meaningful OR-search terms."""
    terms: list[str] = []
    suffixes = ("에서", "으로", "에게", "부터", "까지", "인데", "는데", "이", "가", "은", "는", "을", "를")
    for value in values or []:
        normalized = normalize_korean_text(value)
        if not normalized:
            continue
        terms.append(normalized)
        for token in normalized.split():
            if len(token) < 2 or token in SEARCH_STOPWORDS:
                continue
            terms.append(token)
            for suffix in suffixes:
                if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                    terms.append(token[: -len(suffix)])
                    break
    return list(dict.fromkeys(terms))
