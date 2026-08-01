# Finance RAG — FinanceBench (ICAIF-24)

Retrieval-Augmented Generation для финансовых документов.

Датасет: [ACM-ICAIF '24 Finance RAG Challenge](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge) — бенчмарк **FinanceBench**.

Формат данных (как в соревновании):

| File | Description |
|------|-------------|
| `corpus.jsonl` | пассажи/документы: `_id`, `title`, `text` |
| `queries.jsonl` | вопросы: `_id`, `text` |
| `FinanceBench_qrels.tsv` | `query_id`, `corpus_id`, `score` |

Альтернатива: [Linq-AI-Research/FinanceRAG](https://huggingface.co/datasets/Linq-AI-Research/FinanceRAG) на HuggingFace.

## Пайплайн

1. **Sparse** — BM25 (`rank_bm25`)
2. **Dense** — `sentence-transformers` + FAISS (numpy fallback)
3. **Hybrid** — Reciprocal Rank Fusion (RRF)
4. **Rerank** — cross-encoder (`ms-marco-MiniLM-L-6-v2`), опционально
5. **Generate** — Groq API / Ollama / extractive fallback, с цитированием `[1]`, `[2]`
6. **Eval** — NDCG@k, Recall@k, MRR@k по qrels

## Структура

```
.
├── data/                  # corpus.jsonl, queries.jsonl, *qrels*.tsv
├── notebooks/
│   └── finance_rag.ipynb
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
├── models/                # кэш эмбеддингов (опционально)
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

Для генерации:

```bash
export GROQ_API_KEY=...          # https://console.groq.com
# или
ollama pull llama3.1 && ollama serve
```

## Данные

```bash
mkdir -p data
# из Kaggle competition data или HF — положите:
#   data/corpus.jsonl
#   data/queries.jsonl
#   data/FinanceBench_qrels.tsv
```

Без этих файлов `load_financebench()` падает с `FileNotFoundError` (синтетика не подставляется).

## Запуск

```bash
jupyter notebook notebooks/finance_rag.ipynb
```

Ноутбук:
- загружает corpus / queries / qrels
- строит BM25 + dense индекс
- сравнивает dense / bm25 / hybrid / hybrid+reranker по NDCG@10
- генерирует ответы на нескольких вопросах с цитатами
- сохраняет retrieval results в `results/`

## Результаты

На FinanceBench порядок обычно такой (зависит от модели эмбеддингов и размера корпуса):

| Модель            | ndcg@5 | recall@5 | mrr@5 | ndcg@10 | recall@10 | mrr@10 |
|-------------------|--------|----------|-------|---------|-----------|--------|
| dense             | 0.7193 | 0.9222   | 0.6785 | 0.7424  | 0.9889    | 0.6841 |
| hybrid+rerank     | 0.6859 | 0.8444   | 0.6548 | 0.7011  | 0.8889    | 0.6570 |
| hybrid            | 0.4642 | 0.6222   | 0.4233 | 0.4888  | 0.6889    | 0.4344 |
| bm25              | 0.2010 | 0.2444   | 0.2026 | 0.2357  | 0.3444    | 0.2187 |

## Что можно улучшить

- Domain embedding (`BAAI/bge-base-en-v1.5`, finance-tuned models) вместо MiniLM
- Chunking длинных 10-K по секциям (Item 1A, MD&A) вместо готовых пассажей
- Query expansion / HyDE перед ретривалом
- RAGAS (faithfulness, answer_relevancy) — нужен LLM-as-judge, дорого по токенам
- Late interaction (ColBERT) для длинных финансовых текстов
- Кэш dense-индекса на диск между запусками

## Воспроизводимость

`RANDOM_STATE = 42`. Версии — в `requirements.txt`.
