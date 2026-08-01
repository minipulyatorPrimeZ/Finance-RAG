"""Hybrid fusion: Reciprocal Rank Fusion (RRF) и weighted sum."""

from __future__ import annotations

from collections import defaultdict


def rrf_fuse(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int = 50,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion.
    rankings — список ранжирований [(doc_id, score), ...] уже отсортированных desc.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


def weighted_fuse(
    rankings: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
    top_k: int = 50,
) -> list[tuple[str, float]]:
    """
    Min-max нормализация скоров внутри каждого ranking + weighted sum.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    assert len(weights) == len(rankings)

    scores: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        if not ranking:
            continue
        vals = [s for _, s in ranking]
        lo, hi = min(vals), max(vals)
        denom = (hi - lo) if hi > lo else 1.0
        for doc_id, s in ranking:
            scores[doc_id] += w * ((s - lo) / denom)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


def hybrid_search(
    dense_hits: list[tuple[str, float]],
    sparse_hits: list[tuple[str, float]],
    method: str = "rrf",
    top_k: int = 50,
    dense_weight: float = 0.5,
) -> list[tuple[str, float]]:
    if method == "rrf":
        return rrf_fuse([dense_hits, sparse_hits], top_k=top_k)
    return weighted_fuse(
        [dense_hits, sparse_hits],
        weights=[dense_weight, 1.0 - dense_weight],
        top_k=top_k,
    )
