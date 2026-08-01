"""Cross-encoder reranking."""

from __future__ import annotations

from .data_loader import doc_text

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        corpus: dict[str, dict],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = []
        ids = []
        for doc_id, _ in candidates:
            if doc_id not in corpus:
                continue
            pairs.append([query, doc_text(corpus[doc_id], max_chars=2000)])
            ids.append(doc_id)
        if not pairs:
            return []
        scores = self.model.predict(pairs)
        ranked = sorted(zip(ids, scores), key=lambda x: float(x[1]), reverse=True)
        return [(d, float(s)) for d, s in ranked[:top_k]]
