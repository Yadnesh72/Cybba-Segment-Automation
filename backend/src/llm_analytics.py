# backend/src/llm_analytics.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional

import requests

OLLAMA_URL_DEFAULT = "http://localhost:11434/api/generate"


def _clean(s: object) -> str:
    return str(s or "").strip()


def _clean_insights_text(text: str) -> str:
    """
    Keep it readable, remove obvious junk, but allow multi-line output.
    """
    t = _clean(text)

    # remove common preambles
    t = re.sub(r"^(sure[,!]\s*|here(?:'s| is)\s+).*?\n", "", t, flags=re.IGNORECASE)

    return t.strip()


def _build_prompt(payload: Dict[str, Any]) -> str:
    chart_id = _clean(payload.get("chartId"))
    metric_label = _clean(payload.get("metricLabel") or payload.get("metric"))

    totals = payload.get("totals") or {}
    stats = payload.get("stats") or {}
    sample = payload.get("sample") or {}

    # chart-specific slices (all optional)
    top_high = sample.get("topHigh") or []
    top_low = sample.get("topLow") or []
    cat_summary = sample.get("catPriceSummary") or []
    top_categories = sample.get("topCategories") or []
    top_reused = sample.get("topReused") or []
    price_buckets = sample.get("priceBuckets") or []
    scatter = sample.get("scatter") or []
    reuse_depth = sample.get("reuseDepth") or []   # only if you add it later

    base = f"""
You are helping interpret analytics charts for Cybba Segment Expansion.

CRITICAL RULES:
- Use ONLY the provided numbers/lists (and the image). If a metric/value is not present, DO NOT mention it.
- Do NOT invent fields like "reuse depth" unless it is explicitly provided in the payload.
- When you cite a number, it must exist in totals/stats/sample OR be visually obvious from the chart.
- Keep it actionable: what to audit, what to validate, what might be wrong.

Chart: {chart_id}
Metric: {metric_label}

Totals:
{totals}

Stats:
{stats}
""".strip()

    # ✅ chart-specific instructions
    if chart_id == "priceBuckets":
        specific = f"""
This is a QUARTILE DONUT (4 buckets). Focus on:
- how balanced or skewed the quartiles are
- whether the top quartile is too wide (possible pricing inflation)
- identify true high outliers using topHigh/topLow lists (if provided)

Outliers (highest prices):
{top_high}

Outliers (lowest prices):
{top_low}
""".strip()

    elif chart_id == "categoryDist":
        specific = f"""
This is a CATEGORY DISTRIBUTION PIE. Focus on:
- concentration: whether 1 category dominates
- whether "Other" or "Uncategorized" is large (taxonomy issues)
- suggest what to check in taxonomy mapping logic

Top categories:
{top_categories}
""".strip()

    elif chart_id == "reuseDepth":
        specific = f"""
This is a REUSE DEPTH BAR chart (sources per segment). Focus on:
- how many segments are 0/1 source vs 5+
- whether heavy compositions correlate with price outliers (ONLY if topHigh/topLow are provided)

Outliers (highest prices):
{top_high}

Outliers (lowest prices):
{top_low}
""".strip()

    elif chart_id == "priceDist":
        specific = f"""
This is a PRICE DISTRIBUTION (bucketed curve). Focus on:
- shape (spikes, long tail)
- whether pricing is clustered unnaturally (binning artifacts or model collapse)
- outlier tails

Bucket sample:
{price_buckets}

Outliers (highest prices):
{top_high}

Outliers (lowest prices):
{top_low}
""".strip()

    elif chart_id == "scatter":
        specific = f"""
This is a SCATTER of uniqueness vs rank. Focus on:
- clusters and quadrants
- identify top-right candidates (high rank + high uniqueness)
- if correlation is weak/strong, interpret it using provided corr (if present)

Scatter sample:
{scatter}
""".strip()

    elif chart_id == "topReused":
        specific = f"""
This is a MOST REUSED SOURCES bar chart. Focus on:
- which sources dominate compositions
- whether a few sources create repeated patterns (risk of overfitting/duplication)

Top reused sources:
{top_reused}
""".strip()

    else:
        specific = "Focus on what this chart shows. If unsure, describe what you can infer from the image without inventing data."

    return f"""{base}

{specific}

Return in this exact format:

Takeaway: <one sentence>

Insights:
- <bullet 1 with names/values when available>
- <bullet 2>
- <bullet 3 optional>

Checks:
- <bullet 1>
- <bullet 2 optional>
""".strip()


def generate_analytics_insights(
    payload: Dict[str, Any],
    *,
    # IMPORTANT: must be a vision model in Ollama
    model: str = "llava",
    ollama_url: str = OLLAMA_URL_DEFAULT,
    timeout: int = 120,
) -> str:
    prompt = _build_prompt(payload)

    # chart image from frontend (base64, NO data:image/... prefix)
    chart_b64 = _clean(payload.get("chartImageB64") or payload.get("chart_image_b64") or "")

    req = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 220,
        },
    }

    if chart_b64:
        req["images"] = [chart_b64]

        

    try:
        print(f"[LLM_ANALYTICS] POST {ollama_url} model={model} has_image={bool(chart_b64)}")
        resp = requests.post(ollama_url, json=req, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        raw = _clean(data.get("response", ""))
        return _clean_insights_text(raw)
    except Exception as e:
        print(f"[LLM_ANALYTICS] error: {type(e).__name__}: {e}")
        return ""