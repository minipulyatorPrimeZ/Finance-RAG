# Finance RAG — FinanceBench (ICAIF-24)

Retrieval-Augmented Generation для финансовых документов.

Датасет: [ACM-ICAIF '24 Finance RAG Challenge](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge) — бенчмарк **FinanceBench**.

Формат данных:

| File | Description |
|------|-------------|
| `corpus.jsonl` | `_id`, `title`, `text` |
| `queries.jsonl` | `_id`, `text` |
| `FinanceBench_qrels.tsv` | `query_id`, `corpus_id`, `score` |

## Версии / Changelog

### v2
- Reranker по умолчанию: `BAAI/bge-reranker-v2-m3` (вместо ms-marco MiniLM)
- В генерацию уходит **5–7** пассажей (default 7), не 3
- Structured answer: `MODE` = exact | inferred | insufficient
- Если точного факта нет — модель описывает, что нашлось, и делает осторожный вывод (inference)
- **Confidence score** (0–1): смесь LLM self-report + сила retrieval/rerank scores
- `format_answer()` для читаемого вывода
- Reranker включён по умолчанию в `RetrievalPipeline`

### v1
- BM25 + dense (MiniLM) + RRF hybrid
- Optional CE rerank (ms-marco MiniLM)
- Generation: Groq / Ollama / extractive
- Eval: NDCG@k, Recall@k, MRR@k

## Пайплайн (v2)

1. **Sparse** — BM25
2. **Dense** — `sentence-transformers` + FAISS (numpy fallback)
3. **Hybrid** — RRF
4. **Rerank** — `BAAI/bge-reranker-v2-m3`
5. **Generate** — Groq / Ollama, 5–7 чанков, citations, confidence
6. **Eval** — NDCG@k / Recall@k / MRR@k

## Структура

```
.
├── data/
├── notebooks/finance_rag.ipynb
├── src/
│   ├── data_loader.py
│   ├── sparse.py
│   ├── dense.py
│   ├── hybrid.py
│   ├── rerank.py
│   ├── generate.py
│   ├── evaluate.py
│   ├── pipeline.py
│   └── utils.py
├── models/
├── results/
├── requirements.txt
└── README.md
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Первый запуск скачает embedding + reranker (bge-reranker-v2-m3 заметно тяжелее MiniLM).

```bash
export GROQ_API_KEY=...     # https://console.groq.com
# или
ollama pull llama3.1 && ollama serve
```

## Данные

```bash
mkdir -p data
# corpus.jsonl, queries.jsonl, FinanceBench_qrels.tsv → data/
```

Без файлов `load_financebench()` поднимает `FileNotFoundError`.

## Запуск

```bash
jupyter notebook notebooks/finance_rag.ipynb
```

## Формат ответа (v2)

```
MODE: exact | inferred | insufficient
CONFIDENCE: 0.82 (retrieval=0.71, llm=0.90)

<answer text>

FOUND:
...

REASONING:
...   # если inferred / insufficient

SOURCES:
doc_id_1, doc_id_2, ...
```

Confidence:
- high (≥0.85) — ответ прямо в контексте
- mid (0.5–0.84) — частичная поддержка / inference
- low (<0.5) — данных мало, ответу нельзя доверять как факту

## Результаты

Сравнение методов по метрикам NDCG, Recall и MRR на тестовой выборке:

| Метод              | ndcg@5 | recall@5 | mrr@5 | ndcg@10 | recall@10 | mrr@10 |
|--------------------|--------|----------|-------|---------|-----------|--------|
| **hybrid+rerank**  | 0.8586 | 0.9444   | 0.8478| 0.8660  | 0.9667    | 0.8510 |
| dense              | 0.7193 | 0.9222   | 0.6785| 0.7241  | 0.9333    | 0.6785 |
| hybrid             | 0.4642 | 0.6222   | 0.4233| 0.4849  | 0.6778    | 0.4344 |
| bm25               | 0.2010 | 0.2444   | 0.2026| 0.2055  | 0.2556    | 0.2058 |

## Что ещё можно улучшить

- Domain embedding (`BAAI/bge-base-en-v1.5`) вместо MiniLM
- Section-aware chunking 10-K
- Query expansion / HyDE
- RAGAS (faithfulness, answer_relevancy)
- Кэш FAISS + reranker weights на диск
- FIXME: для очень длинных пассажей bge-reranker режет по max_length=512 — иногда теряется хвост таблицы

## Воспроизводимость

`RANDOM_STATE = 42`. Зависимости — `requirements.txt`.
