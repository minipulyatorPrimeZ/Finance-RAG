"""Retrieval metrics: NDCG, Recall, MRR."""

from __future__ import annotations

import math
from typing import Any


def _dcg(relevances: list[float], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(relevances[:k]):
        s += (2**rel - 1) / math.log2(i + 2)
    return s


def ndcg_at_k(
    ranked_ids: list[str],
    qrels: dict[str, int],
    k: int = 10,
) -> float:
    """NDCG@k для одного запроса. qrels: doc_id -> relevance (>=1 relevant)."""
    if not qrels:
        return 0.0
    gains = [float(qrels.get(d, 0)) for d in ranked_ids[:k]]
    ideal = sorted(qrels.values(), reverse=True)
    idcg = _dcg([float(x) for x in ideal], k)
    if idcg == 0:
        return 0.0
    return _dcg(gains, k) / idcg


def recall_at_k(
    ranked_ids: list[str],
    qrels: dict[str, int],
    k: int = 10,
) -> float:
    relevant = {d for d, s in qrels.items() if s > 0}
    if not relevant:
        return 0.0
    hit = sum(1 for d in ranked_ids[:k] if d in relevant)
    return hit / len(relevant)


def mrr_at_k(
    ranked_ids: list[str],
    qrels: dict[str, int],
    k: int = 10,
) -> float:
    relevant = {d for d, s in qrels.items() if s > 0}
    for i, d in enumerate(ranked_ids[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(
    results: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    results: query_id -> [(doc_id, score), ...]
    qrels: query_id -> {doc_id: score}
    """
    if k_values is None:
        k_values = [5, 10]

    metrics: dict[str, list[float]] = {}
    for k in k_values:
        metrics[f"ndcg@{k}"] = []
        metrics[f"recall@{k}"] = []
        metrics[f"mrr@{k}"] = []

    n = 0
    for qid, ranking in results.items():
        if qid not in qrels:
            continue
        rel = qrels[qid]
        ranked_ids = [d for d, _ in ranking]
        n += 1
        for k in k_values:
            metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked_ids, rel, k))
            metrics[f"recall@{k}"].append(recall_at_k(ranked_ids, rel, k))
            metrics[f"mrr@{k}"].append(mrr_at_k(ranked_ids, rel, k))

    if n == 0:
        raise ValueError("Нет пересечения query_id между results и qrels")

    return {name: sum(vals) / len(vals) for name, vals in metrics.items()}


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"{name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
