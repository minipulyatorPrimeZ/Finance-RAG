"""Сборка end-to-end retrieval pipeline (v2)."""

from __future__ import annotations

from typing import Literal

from .dense import DenseRetriever
from .hybrid import hybrid_search
from .rerank import CrossEncoderReranker, DEFAULT_RERANKER
from .sparse import BM25Retriever


class RetrievalPipeline:
    def __init__(
        self,
        corpus: dict[str, dict],
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_reranker: bool = True,
        reranker_model: str = DEFAULT_RERANKER,
        device: str | None = None,
    ):
        self.corpus = corpus
        print(f"Building BM25 over {len(corpus)} docs...")
        self.bm25 = BM25Retriever(corpus)
        print(f"Encoding dense index ({dense_model})...")
        self.dense = DenseRetriever(corpus, model_name=dense_model, device=device)
        self.reranker = None
        if use_reranker:
            print(f"Loading reranker ({reranker_model})...")
            # TODO: кэшировать веса на диск, чтобы не качать при каждом рестарте
            self.reranker = CrossEncoderReranker(reranker_model, device=device)

    def retrieve(
        self,
        query: str,
        mode: Literal["dense", "bm25", "hybrid"] = "hybrid",
        top_k: int = 50,
        rerank_top_k: int = 7,
        fusion: str = "rrf",
    ) -> list[tuple[str, float]]:
        """
        top_k — кандидаты до реранка
        rerank_top_k — сколько вернуть после (v2 default 7)
        """
        if mode == "dense":
            hits = self.dense.search(query, top_k=top_k)
        elif mode == "bm25":
            hits = self.bm25.search(query, top_k=top_k)
        else:
            d_hits = self.dense.search(query, top_k=top_k)
            s_hits = self.bm25.search(query, top_k=top_k)
            hits = hybrid_search(d_hits, s_hits, method=fusion, top_k=top_k)

        if self.reranker is not None:
            hits = self.reranker.rerank(
                query, hits[:top_k], self.corpus, top_k=rerank_top_k
            )
        else:
            hits = hits[:rerank_top_k]
        return hits

    def retrieve_many(
        self,
        queries: dict[str, str],
        mode: str = "hybrid",
        top_k: int = 50,
        rerank_top_k: int = 7,
        fusion: str = "rrf",
    ) -> dict[str, list[tuple[str, float]]]:
        out = {}
        n = len(queries)
        for i, (qid, qtext) in enumerate(queries.items()):
            if (i + 1) % 20 == 0 or (i + 1) == n:
                print(f"  retrieve {i + 1}/{n}")
            out[qid] = self.retrieve(
                qtext,
                mode=mode,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                fusion=fusion,
            )
        return out
