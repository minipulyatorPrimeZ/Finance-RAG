"""
Генерация ответа с цитированием, inference-режимом и confidence (v2).

Бэкенды:
  - groq  (GROQ_API_KEY)
  - ollama (локально)
  - extractive — без LLM
"""

from __future__ import annotations

import os
import re
from typing import Any

from .data_loader import doc_text
from .utils import truncate


SYSTEM_PROMPT = """You are a financial analyst assistant answering questions over SEC filings and related documents.

Rules:
1. Prefer facts that appear explicitly in the context. Cite them as [1], [2], ... using passage numbers.
2. If the exact answer is NOT in the context, do NOT invent numbers or events.
   Instead:
   - summarize what relevant information WAS found;
   - state clearly that the exact figure/fact is missing;
   - if a cautious logical inference is possible from the found data, give it and label it as inference.
3. Always end with a confidence line: CONFIDENCE: <float 0.0-1.0>
   - 0.85–1.0: answer is directly supported by context
   - 0.50–0.84: partial support / reasonable inference
   - 0.0–0.49: weak or no support
4. Be concise. Keep numbers and units exact when quoting the text.
5. Output format:

ANSWER: <main answer or short statement>
MODE: exact | inferred | insufficient
FOUND: <what relevant facts were retrieved, 1-3 bullets or one short paragraph>
REASONING: <only if MODE is inferred or insufficient; otherwise "n/a">
CONFIDENCE: <0.00-1.00>
SOURCES: <comma-separated passage numbers used, e.g. 1,3>
"""


def build_context(
    hits: list[tuple[str, float]],
    corpus: dict[str, dict],
    max_passages: int = 7,
    max_chars_per_passage: int = 1800,
) -> tuple[str, list[str]]:
    """Контекст + список doc_id в порядке цитирования."""
    parts = []
    used_ids = []
    for i, (doc_id, score) in enumerate(hits[:max_passages], start=1):
        if doc_id not in corpus:
            continue
        text = truncate(doc_text(corpus[doc_id]), max_chars_per_passage)
        title = corpus[doc_id].get("title") or doc_id
        parts.append(f"[{i}] (id={doc_id}, title={title}, score={score:.4f})\n{text}")
        used_ids.append(doc_id)
    return "\n\n".join(parts), used_ids


def build_messages(query: str, context: str) -> list[dict[str, str]]:
    user = (
        f"Context passages:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Respond in the required format."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _retrieval_confidence(hits: list[tuple[str, float]], top_n: int = 5) -> float:
    """
    Грубая оценка по реранк/ретривал скорам.
    CE scores (bge-reranker) часто в широком диапазоне — min-max по top_n.
    """
    if not hits:
        return 0.0
    scores = [float(s) for _, s in hits[:top_n]]
    # sigmoid-ish нормализация для CE logits
    import math

    def sig(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    # если скоры уже в [0,1] (cosine) — берём mean top
    if all(0.0 <= s <= 1.0 for s in scores):
        return float(sum(scores) / len(scores))
    # иначе считаем как logits
    return float(sum(sig(s) for s in scores) / len(scores))


def parse_structured_answer(raw: str) -> dict[str, Any]:
    """Парсит ANSWER/MODE/FOUND/REASONING/CONFIDENCE из ответа модели."""
    def _field(name: str) -> str | None:
        m = re.search(
            rf"(?is)^{name}\s*:\s*(.+?)(?=^(?:ANSWER|MODE|FOUND|REASONING|CONFIDENCE|SOURCES)\s*:|\Z)",
            raw,
            re.MULTILINE,
        )
        return m.group(1).strip() if m else None

    answer = _field("ANSWER") or raw.strip()
    mode = (_field("MODE") or "exact").lower()
    if mode not in ("exact", "inferred", "insufficient"):
        # эвристика
        low = raw.lower()
        if "insufficient" in low or "not found" in low or "не найден" in low:
            mode = "insufficient"
        elif "infer" in low or "based on" in low:
            mode = "inferred"
        else:
            mode = "exact"

    found = _field("FOUND") or ""
    reasoning = _field("REASONING") or ""
    conf_raw = _field("CONFIDENCE")
    llm_conf = None
    if conf_raw:
        m = re.search(r"([01](?:\.\d+)?)", conf_raw)
        if m:
            llm_conf = float(m.group(1))
            llm_conf = max(0.0, min(1.0, llm_conf))

    return {
        "answer": answer,
        "mode": mode,
        "found": found,
        "reasoning": reasoning,
        "llm_confidence": llm_conf,
    }


def combine_confidence(
    llm_conf: float | None,
    retrieval_conf: float,
    mode: str,
) -> float:
    """
    Итоговый confidence: смесь self-report модели и силы ретривала.
    MODE=insufficient жёстко даунскейлит.
    """
    if llm_conf is None:
        base = retrieval_conf
    else:
        base = 0.6 * llm_conf + 0.4 * retrieval_conf

    if mode == "insufficient":
        base = min(base, 0.45)
    elif mode == "inferred":
        base = min(base, 0.75)

    return round(float(max(0.0, min(1.0, base))), 3)


def generate_extractive(
    query: str,
    hits: list[tuple[str, float]],
    corpus: dict[str, dict],
    max_passages: int = 7,
) -> dict[str, Any]:
    context, ids = build_context(hits, corpus, max_passages=max_passages)
    ret_c = _retrieval_confidence(hits)
    answer = (
        "LLM backend unavailable (set GROQ_API_KEY or run Ollama).\n"
        "Showing top retrieved passages instead.\n\n"
        + context
    )
    return {
        "answer": answer,
        "mode": "insufficient",
        "found": f"Retrieved {len(ids)} passages (extractive fallback).",
        "reasoning": "n/a — no LLM",
        "confidence": round(min(ret_c, 0.4), 3),
        "retrieval_confidence": round(ret_c, 3),
        "llm_confidence": None,
        "source_ids": ids,
        "backend": "extractive",
        "raw": answer,
    }


def _call_chat(messages: list[dict], backend: str, model: str, **kw) -> str:
    if backend == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        from groq import Groq

        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=kw.get("max_tokens", 700),
        )
        return resp.choices[0].message.content or ""

    if backend == "ollama":
        import json
        import urllib.request

        base_url = kw.get("base_url", "http://localhost:11434")
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("message") or {}).get("content") or ""

    raise ValueError(f"unknown backend: {backend}")


def _pack_result(
    raw: str,
    hits: list[tuple[str, float]],
    source_ids: list[str],
    backend: str,
) -> dict[str, Any]:
    parsed = parse_structured_answer(raw)
    ret_c = _retrieval_confidence(hits)
    conf = combine_confidence(parsed["llm_confidence"], ret_c, parsed["mode"])
    return {
        "answer": parsed["answer"],
        "mode": parsed["mode"],
        "found": parsed["found"],
        "reasoning": parsed["reasoning"],
        "confidence": conf,
        "retrieval_confidence": round(ret_c, 3),
        "llm_confidence": parsed["llm_confidence"],
        "source_ids": source_ids,
        "backend": backend,
        "raw": raw,
    }


def generate_groq(
    query: str,
    hits: list[tuple[str, float]],
    corpus: dict[str, dict],
    model: str = "llama-3.1-8b-instant",
    max_passages: int = 7,
) -> dict[str, Any]:
    context, ids = build_context(hits, corpus, max_passages=max_passages)
    raw = _call_chat(build_messages(query, context), "groq", model)
    return _pack_result(raw, hits, ids, f"groq:{model}")


def generate_ollama(
    query: str,
    hits: list[tuple[str, float]],
    corpus: dict[str, dict],
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    max_passages: int = 7,
) -> dict[str, Any]:
    context, ids = build_context(hits, corpus, max_passages=max_passages)
    raw = _call_chat(
        build_messages(query, context), "ollama", model, base_url=base_url
    )
    return _pack_result(raw, hits, ids, f"ollama:{model}")


def generate_answer(
    query: str,
    hits: list[tuple[str, float]],
    corpus: dict[str, dict],
    backend: str = "auto",
    max_passages: int = 7,
    **kwargs,
) -> dict[str, Any]:
    """
    backend: auto | groq | ollama | extractive
    max_passages: 5–7 рекомендуется в v2
    """
    kwargs = {**kwargs, "max_passages": max_passages}

    if backend == "extractive":
        return generate_extractive(query, hits, corpus, max_passages=max_passages)
    if backend == "groq":
        return generate_groq(query, hits, corpus, **kwargs)
    if backend == "ollama":
        return generate_ollama(query, hits, corpus, **kwargs)

    if os.environ.get("GROQ_API_KEY"):
        try:
            return generate_groq(query, hits, corpus, **kwargs)
        except Exception as e:
            print(f"groq failed: {e}, trying ollama...")
    try:
        return generate_ollama(query, hits, corpus, **kwargs)
    except Exception as e:
        print(f"ollama failed: {e}, falling back to extractive")
        return generate_extractive(query, hits, corpus, max_passages=max_passages)


def format_answer(result: dict[str, Any]) -> str:
    """Человекочитаемый вывод для ноутбука / CLI."""
    lines = [
        f"MODE: {result.get('mode')}",
        f"CONFIDENCE: {result.get('confidence')} "
        f"(retrieval={result.get('retrieval_confidence')}, "
        f"llm={result.get('llm_confidence')})",
        f"BACKEND: {result.get('backend')}",
        "",
        str(result.get("answer") or "").strip(),
    ]
    if result.get("found"):
        lines += ["", "FOUND:", str(result["found"]).strip()]
    if result.get("reasoning") and str(result["reasoning"]).lower() not in ("n/a", "na", ""):
        lines += ["", "REASONING:", str(result["reasoning"]).strip()]
    if result.get("source_ids"):
        lines += ["", "SOURCES:", ", ".join(result["source_ids"])]
    return "\n".join(lines)
