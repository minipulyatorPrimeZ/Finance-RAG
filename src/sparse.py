"""BM25 sparse retrieval."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

from .data_loader import doc_text


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Retriever:
    def __init__(self, corpus: dict[str, dict]):
        self.doc_ids = list(corpus.keys())
        self.corpus = corpus
        tokenized = [tokenize(doc_text(corpus[d])) for d in self.doc_ids]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # top-k без полной сортировки всего корпуса
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in idx if scores[i] > 0]
