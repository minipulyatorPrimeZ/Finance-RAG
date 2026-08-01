"""
Загрузка FinanceBench / ICAIF-24 формата.

Ожидаемые файлы в data_dir (или data_dir/FinanceBench/):
  corpus.jsonl   — {_id, title?, text}
  queries.jsonl  — {_id, text}  (иногда title вместо text)
  *qrels*.tsv    — query_id \\t corpus_id \\t score
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import read_jsonl


def _find_file(data_dir: Path, names: list[str]) -> Path | None:
    for n in names:
        p = data_dir / n
        if p.exists():
            return p
    # один уровень вложенности (FinanceBench/)
    if data_dir.is_dir():
        for sub in data_dir.iterdir():
            if sub.is_dir():
                for n in names:
                    p = sub / n
                    if p.exists():
                        return p
    return None


def load_corpus(data_dir: str | Path) -> dict[str, dict]:
    """
    Returns
    -------
    dict: doc_id -> {title, text, raw}
    """
    data_dir = Path(data_dir)
    path = _find_file(data_dir, ["corpus.jsonl", "FinanceBench_corpus.jsonl"])
    if path is None:
        raise FileNotFoundError(
            f"corpus.jsonl не найден в {data_dir}. "
            "Скачайте FinanceBench из ICAIF-24 (Kaggle) или "
            "Linq-AI-Research/FinanceRAG и положите corpus.jsonl в data/."
        )
    rows = read_jsonl(path)
    corpus = {}
    for r in rows:
        doc_id = str(r.get("_id") or r.get("id") or r.get("corpus_id"))
        title = r.get("title") or r.get("doc_name") or ""
        text = r.get("text") or r.get("contents") or r.get("content") or ""
        if not text and title:
            text = title
        corpus[doc_id] = {"title": title, "text": text, "raw": r}
    if not corpus:
        raise ValueError(f"Пустой corpus: {path}")
    return corpus


def load_queries(data_dir: str | Path) -> dict[str, str]:
    """query_id -> query text"""
    data_dir = Path(data_dir)
    path = _find_file(data_dir, ["queries.jsonl", "FinanceBench_queries.jsonl"])
    if path is None:
        raise FileNotFoundError(
            f"queries.jsonl не найден в {data_dir}. "
            "Ожидается ICAIF-24 / FinanceBench формат."
        )
    rows = read_jsonl(path)
    queries = {}
    for r in rows:
        qid = str(r.get("_id") or r.get("id") or r.get("query_id"))
        text = r.get("text") or r.get("query") or r.get("title") or ""
        queries[qid] = text
    if not queries:
        raise ValueError(f"Пустой queries: {path}")
    return queries


def load_qrels(data_dir: str | Path) -> dict[str, dict[str, int]]:
    """
    Returns
    -------
    dict: query_id -> {corpus_id: relevance_score}
    """
    data_dir = Path(data_dir)
    path = _find_file(
        data_dir,
        [
            "FinanceBench_qrels.tsv",
            "qrels.tsv",
            "financebench_qrels.tsv",
            "qrels.test.tsv",
        ],
    )
    if path is None:
        raise FileNotFoundError(
            f"qrels.tsv не найден в {data_dir}. "
            "Нужен файл вида query_id\\tcorpus_id\\tscore."
        )
    df = pd.read_csv(path, sep="\t")
    # гибкие имена колонок
    cols = {c.lower().replace("-", "_"): c for c in df.columns}
    qcol = cols.get("query_id") or cols.get("qid") or list(df.columns)[0]
    dcol = cols.get("corpus_id") or cols.get("doc_id") or list(df.columns)[1]
    scol = cols.get("score") or cols.get("relevance") or (
        list(df.columns)[2] if len(df.columns) > 2 else None
    )

    qrels: dict[str, dict[str, int]] = {}
    for _, row in df.iterrows():
        qid = str(row[qcol])
        did = str(row[dcol])
        score = int(row[scol]) if scol is not None else 1
        qrels.setdefault(qid, {})[did] = score
    return qrels


def load_financebench(data_dir: str | Path = "data"):
    """Удобная обёртка: corpus, queries, qrels."""
    corpus = load_corpus(data_dir)
    queries = load_queries(data_dir)
    qrels = load_qrels(data_dir)
    return corpus, queries, qrels


def doc_text(doc: dict, max_chars: int | None = None) -> str:
    """title + text для индексации / промпта."""
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title and text:
        full = f"{title}\n{text}"
    else:
        full = title or text
    if max_chars is not None and len(full) > max_chars:
        full = full[: max_chars - 3] + "..."
    return full
