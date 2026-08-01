"""
Генерация ответа с цитированием источников.

Бэкенды:
  - groq  (GROQ_API_KEY)
  - ollama (локально, http://localhost:11434)
  - extractive — без LLM, склеивает топ-пассажи (fallback)
"""

from __future__ import annotations

import os
from typing import Any

from .data_loader import doc_text
from .utils import truncate


SYSTEM_PROMPT = """You are a financial analyst assistant.
Answer the question using ONLY the provided context passages.
If the context is insufficient, say so explicitly.
Cite sources as [1], [2], ... matching the passage numbers.
Be concise and precise with numbers and units."""


def build_context(
    hits: list[tuple[str, float]],
    corpus: dict[str, dict],
    max_passages: int = 5,
    max_chars_per_passage: int = 1500,
) -> tuple[str, list[str]]:
    """Возвращает текст контекста и список doc_id в порядке цитирования."""
    parts = []
    used_ids = []
    for i, (doc_id, score) in enumerate(hits[:max_passages], start=1):
        if doc_id not in corpus:
            continue
        text = truncate(doc_text(corpus[doc_id]), max_chars_per_passage)
        title = corpus[doc_id].get("title") or doc_id
        parts.append(f"[{i}] ({title})\n{text}")
        used_ids.append(doc_id)
    return "\n\n".join(parts), used_ids


def build_messages(query: str, context: str) -> list[dict[str, str]]:
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_extractive(
    query: str,
    hits: list[tuple[str, float]],
    corpus: dict[str, dict],
    max_passages: int = 3,
) -> dict[str, Any]:
    """Fallback без LLM — просто возвращает склеенный контекст."""
    context, ids = build_context(hits, corpus, max_passages=max_passages)
    answer = (
        "[extractive fallback — set GROQ_API_KEY or run Ollama for generation]\n\n"
        + context
    )
    return {"answer": answer, "source_ids": ids, "backend": "extractive"}


def generate_groq(
    query: str,
    hits: list[tuple[str, float]],
    corpus: dict[str, dict],
    model: str = "llama-3.1-8b-instant",
    max_passages: int = 5,
) -> dict[str, Any]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    from groq import Groq

    context, ids = build_context(hits, corpus, max_passages=max_passages)
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=build_messages(query, context),
        temperature=0.1,
        max_tokens=512,
    )
    answer = resp.choices[0].message.content or ""
    return {"answer": answer, "source_ids": ids, "backend": f"groq:{model}"}


def generate_ollama(
    query: str,
    hits: list[tuple[str, float]],
    corpus: dict[str, dict],
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    max_passages: int = 5,
) -> dict[str, Any]:
    import urllib.request
    import json

    context, ids = build_context(hits, corpus, max_passages=max_passages)
    payload = {
        "model": model,
        "messages": build_messages(query, context),
        "stream": False,
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e
    answer = (data.get("message") or {}).get("content") or ""
    return {"answer": answer, "source_ids": ids, "backend": f"ollama:{model}"}


def generate_answer(
    query: str,
    hits: list[tuple[str, float]],
    corpus: dict[str, dict],
    backend: str = "auto",
    **kwargs,
) -> dict[str, Any]:
    """
    backend: auto | groq | ollama | extractive
    auto пробует groq → ollama → extractive
    """
    if backend == "extractive":
        return generate_extractive(query, hits, corpus, **kwargs)

    if backend == "groq":
        return generate_groq(query, hits, corpus, **kwargs)

    if backend == "ollama":
        return generate_ollama(query, hits, corpus, **kwargs)

    # auto
    if os.environ.get("GROQ_API_KEY"):
        try:
            return generate_groq(query, hits, corpus, **kwargs)
        except Exception as e:
            print(f"groq failed: {e}, trying ollama...")
    try:
        return generate_ollama(query, hits, corpus, **kwargs)
    except Exception as e:
        print(f"ollama failed: {e}, falling back to extractive")
        return generate_extractive(query, hits, corpus, **kwargs)
