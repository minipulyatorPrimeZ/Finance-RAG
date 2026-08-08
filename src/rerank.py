"""Cross-encoder reranking (v2: BAAI/bge-reranker-v2-m3 by default)."""

from __future__ import annotations

from .data_loader import doc_text

# multilingual, сильнее ms-marco MiniLM на длинных пассажах
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        device: str | None = None,
        max_length: int = 512,
    ):
        from sentence_transformers import CrossEncoder

        # trust_remote_code нужен для части BGE-моделей
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
            trust_remote_code=True,
        )
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        corpus: dict[str, dict],
        top_k: int = 10,
        max_chars: int = 2000,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = []
        ids = []
        for doc_id, _ in candidates:
            if doc_id not in corpus:
                continue
            pairs.append([query, doc_text(corpus[doc_id], max_chars=max_chars)])
            ids.append(doc_id)
        if not pairs:
            return []

        scores = self.model.predict(pairs, show_progress_bar=len(pairs) > 64)
        ranked = sorted(zip(ids, scores), key=lambda x: float(x[1]), reverse=True)
        return [(d, float(s)) for d, s in ranked[:top_k]]
