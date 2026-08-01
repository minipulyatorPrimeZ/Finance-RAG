"""Dense retrieval via sentence-transformers + FAISS (или numpy fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .data_loader import doc_text
from .utils import RANDOM_STATE


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(
        self,
        corpus: dict[str, dict],
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.doc_ids = list(corpus.keys())
        self.corpus = corpus
        self.model = SentenceTransformer(model_name, device=device)
        texts = [doc_text(corpus[d]) for d in self.doc_ids]
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        self._index = None
        self._build_index()

    def _build_index(self) -> None:
        try:
            import faiss

            dim = self.embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self.embeddings)
            self._index = index
            self._backend = "faiss"
        except Exception:
            # fallback: brute-force numpy
            self._index = None
            self._backend = "numpy"

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        if self._backend == "faiss":
            scores, idxs = self._index.search(q, min(top_k, len(self.doc_ids)))
            return [
                (self.doc_ids[int(i)], float(s))
                for i, s in zip(idxs[0], scores[0])
                if i >= 0
            ]
        # numpy
        sims = (self.embeddings @ q[0])
        idx = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.doc_ids[int(i)], float(sims[i])) for i in idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            doc_ids=np.array(self.doc_ids, dtype=object),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        corpus: dict[str, dict],
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> "DenseRetriever":
        """Загружает эмбеддинги с диска, модель — заново (легче чем pickle ST)."""
        from sentence_transformers import SentenceTransformer

        data = np.load(path, allow_pickle=True)
        obj = object.__new__(cls)
        obj.corpus = corpus
        obj.doc_ids = list(data["doc_ids"])
        obj.embeddings = data["embeddings"].astype(np.float32)
        obj.model = SentenceTransformer(model_name, device=device)
        obj._index = None
        obj._build_index()
        return obj
