# backend/src/llm_descriptions.py
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

import pandas as pd
import requests


OLLAMA_URL_DEFAULT = "http://localhost:11434/api/generate"


def _clean(s: object) -> str:
    return str(s or "").strip()


def _stable_key(segment_name: str, components: str) -> str:
    """
    Stable cache key so identical segments don't regenerate.
    """
    base = (_clean(segment_name) + "|" + _clean(components)).encode("utf-8")
    return hashlib.sha256(base).hexdigest()


def _clean_llm_text(text: str) -> str:
    """
    Force the model output to be ONLY the short description sentence.
    Removes chatty prefixes and trims to one line.
    """
    t = _clean(text).strip().strip('"').strip("'")

    # Remove common meta-intros the model might generate
    t = re.sub(
        r"^(here\s+(is|are)\b[^:]*:\s*|here\s+is\b\s*|a\s+concise\b[^:]*:\s*|description\b[^:]*:\s*)",
        "",
        t,
        flags=re.IGNORECASE,
    )

    # Keep first non-empty line only
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    t = lines[0] if lines else ""

    # Remove leading punctuation/quotes again after line selection
    t = t.lstrip(":-–—• ").strip().strip('"').strip("'")

    # Ensure it ends with a period
    if t and not t.endswith((".", "!", "?")):
        t += "."

    return t


def _truncate_words(text: str, max_words: int = 16) -> str:
    """
    Hard cap description length to keep UI fast + consistent.
    """
    t = _clean(text)
    if not t:
        return ""
    words = t.split()
    if len(words) <= max_words:
        return t
    trimmed = " ".join(words[:max_words]).rstrip(",;:-")
    if not trimmed.endswith((".", "!", "?")):
        trimmed += "."
    return trimmed


def _build_prompt(segment_name: str, components: str) -> str:
    """
    Strict prompt: output must start directly with the audience description, no meta text.
    """
    return f"""
Return ONLY the final description sentence. No intro text.

Write ONE short sentence (8–16 words).
Keep it factual and advertiser-friendly.

Segment name:
{segment_name}

Underlying data components (context only):
{components or "N/A"}

Rules:
- Output ONLY the sentence (no labels like "Here is...", no quotes)
- Start directly with the audience description (e.g., "Enthusiasts who ...", "Decision makers at ...")
- Do NOT mention "Cybba"
- Do NOT use bullet points
- Do NOT repeat the full segment name verbatim
- Do NOT invent exact statistics
- End with a period
""".strip()


def generate_segment_description(
    segment_name: str,
    components: Optional[str] = None,
    *,
    model: str = "llama3.1",
    ollama_url: str = OLLAMA_URL_DEFAULT,
    timeout: int = 60,
    max_words: int = 16,
) -> str:
    """
    Generate a single short description using Ollama.
    Designed to be fast + non-chatty.
    """
    name = _clean(segment_name)
    comps = _clean(components)

    if not name:
        return ""

    payload = {
        "model": model,
        "prompt": _build_prompt(name, comps),
        "stream": False,
        "options": {
            # lower creativity for consistent short outputs
            "temperature": 0.2,
            "top_p": 0.9,
            # Ollama uses "num_predict" to limit generated tokens
            # (keeps latency down and prevents rambling)
            "num_predict": 48,
        },
    }

    try:
        resp = requests.post(ollama_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        raw = _clean(data.get("response", ""))
        cleaned = _clean_llm_text(raw)
        return _truncate_words(cleaned, max_words=max_words)
    except Exception:
        # Never break the pipeline if LLM fails
        return ""


def generate_descriptions(
    df: pd.DataFrame,
    *,
    model: str = "llama3.1",
    batch_size: int = 20,
    cache: Optional[Dict[str, str]] = None,
    ollama_url: str = OLLAMA_URL_DEFAULT,
    timeout: int = 60,
    max_words: int = 16,
) -> List[str]:
    """
    Generate descriptions for each row in df, aligned to df.index order.

    Expected columns (best-effort):
      - Proposed New Segment Name (required)
      - Non Derived Segments utilized (optional context)

    Returns:
      list[str] of length len(df)
    """
    if cache is None:
        cache = {}

    if df is None or len(df) == 0:
        return []

    name_col = "Proposed New Segment Name"
    comp_col = "Non Derived Segments utilized"

    if name_col not in df.columns:
        return [""] * len(df)

    names = df[name_col].astype(str).tolist()
    comps = df[comp_col].astype(str).tolist() if comp_col in df.columns else [""] * len(df)

    out: List[str] = [""] * len(df)

    n = len(df)
    step = max(1, int(batch_size))

    for start in range(0, n, step):
        end = min(n, start + step)
        for i in range(start, end):
            segment_name = _clean(names[i])
            components = _clean(comps[i])

            if not segment_name:
                out[i] = ""
                continue

            key = _stable_key(segment_name, components)

            if key in cache:
                out[i] = cache[key]
                continue

            desc = generate_segment_description(
                segment_name,
                components,
                model=model,
                ollama_url=ollama_url,
                timeout=timeout,
                max_words=max_words,
            )

            cache[key] = desc
            out[i] = desc

    return out
