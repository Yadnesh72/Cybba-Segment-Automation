# backend/src/suggestions.py
"""
Smart Suggestions:
1. POST /api/suggestions/generate?run_id=<id>
   - For each segment in a completed run, find matching competitor + Cybba catalog segments
   - Score: high competitor match + low Cybba match = coverage gap = good suggestion
   - Returns top N diverse suggestion items

2. POST /api/suggestions/analyze
   - Takes a segment + its catalog matches
   - Asks Ollama (llama3.1) for an AI analysis: is it useful? scaling tips? competitor context?
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, Query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.pipeline import load_config

router = APIRouter()
logger = logging.getLogger("suggestions")

# ---------------------------------------------------------------------------
# In-memory caches
# ---------------------------------------------------------------------------
_SUGGESTIONS: Dict[str, Dict[str, Any]] = {}

_INDEX_LOCK = threading.Lock()

_CATALOG_INDEX: Dict[str, Any] = {
    "ready": False,
    "comp_vectorizer": None,
    "comp_matrix": None,
    "comp_rows": None,
    "cybba_vectorizer": None,
    "cybba_matrix": None,
    "cybba_rows": None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def _get_base_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_catalog_index(base_dir: Path) -> None:
    cfg = load_config(base_dir)
    input_dir = Path(cfg["paths"]["input_dir"])
    input_a = input_dir / cfg["files"]["input_a"]

    if not input_a.exists():
        logger.warning("Suggestions index: catalog not found at %s", input_a)
        return

    df = pd.read_csv(input_a, dtype=str, keep_default_na=False)

    provider_col = "Provider Name"
    name_col = "Segment Name"
    desc_col = "Segment Description" if "Segment Description" in df.columns else None

    if provider_col not in df.columns or name_col not in df.columns:
        logger.warning("Suggestions index: missing required columns")
        return

    cybba_name = str(cfg.get("provider", {}).get("cybba_name", "Cybba")).strip().lower()

    comp_df = df[df[provider_col].astype(str).str.strip().str.lower() != cybba_name].copy()
    cybba_df = df[df[provider_col].astype(str).str.strip().str.lower() == cybba_name].copy()

    def _build_vec_index(sub_df: pd.DataFrame) -> Tuple[TfidfVectorizer, Any, List[Dict[str, Any]]]:
        texts: List[str] = []
        rows: List[Dict[str, Any]] = []
        for _, r in sub_df.iterrows():
            provider = str(r.get(provider_col, "")).strip() or "Unknown"
            seg = str(r.get(name_col, "")).strip()
            if not seg:
                continue
            desc = str(r.get(desc_col, "")).strip() if desc_col else ""
            text = seg if not desc else f"{seg} | {desc}"
            rows.append({"provider": provider, "segment_name": seg, "description": desc or None})
            texts.append(_normalize(text))

        if not texts:
            return None, None, []

        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, max_features=30000)
        try:
            mat = vec.fit_transform(texts)
        except ValueError:
            vec = TfidfVectorizer(ngram_range=(1, 1), stop_words="english", min_df=1, max_features=30000)
            mat = vec.fit_transform(texts)
        return vec, mat, rows

    comp_vec, comp_mat, comp_rows = _build_vec_index(comp_df)
    cybba_vec, cybba_mat, cybba_rows = _build_vec_index(cybba_df)

    _CATALOG_INDEX.update({
        "ready": True,
        "comp_vectorizer": comp_vec,
        "comp_matrix": comp_mat,
        "comp_rows": comp_rows,
        "cybba_vectorizer": cybba_vec,
        "cybba_matrix": cybba_mat,
        "cybba_rows": cybba_rows,
    })
    logger.info("Suggestions catalog index built: %d competitor, %d Cybba segments",
                len(comp_rows) if comp_rows else 0, len(cybba_rows) if cybba_rows else 0)


def _ensure_index() -> None:
    if _CATALOG_INDEX["ready"]:
        return
    with _INDEX_LOCK:
        # Double-check inside lock in case another thread just built it
        if _CATALOG_INDEX["ready"]:
            return
        try:
            _build_catalog_index(_get_base_dir())
        except Exception as e:
            logger.error("Suggestions: catalog index build failed: %s", e, exc_info=True)


def _match_against(
    segment_name: str,
    vectorizer: Optional[TfidfVectorizer],
    matrix: Any,
    rows: Optional[List[Dict[str, Any]]],
    top_k: int = 5,
    min_sim: float = 0.10,
) -> List[Dict[str, Any]]:
    if vectorizer is None or matrix is None or not rows:
        return []
    try:
        q = _normalize(segment_name)
        qv = vectorizer.transform([q])
        sims = cosine_similarity(qv, matrix).ravel()
        valid = np.where(sims >= min_sim)[0]
        if valid.size == 0:
            return []
        idxs = valid[np.argsort(-sims[valid])][:top_k]
        result = []
        for i in idxs:
            item = dict(rows[int(i)])
            item["similarity"] = round(float(sims[int(i)]), 4)
            result.append(item)
        return result
    except Exception as e:
        logger.warning("Match failed for '%s': %s", segment_name, e)
        return []


def _generate_suggestions_from_rows(
    run_rows: List[Dict[str, Any]],
    top_n: int = 25,
    min_sim: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    For each row in run_rows, find competitor + Cybba matches.
    Score = competitor_max_sim - cybba_max_sim (high gap = good suggestion).
    Return top_n most diverse suggestions.
    """
    _ensure_index()

    suggestions: List[Dict[str, Any]] = []

    for row in run_rows:
        seg_name = str(
            row.get("Proposed New Segment Name") or
            row.get("New Segment Name") or
            row.get("Segment Name") or ""
        ).strip()
        if not seg_name:
            continue

        comp_matches = _match_against(
            seg_name,
            _CATALOG_INDEX["comp_vectorizer"],
            _CATALOG_INDEX["comp_matrix"],
            _CATALOG_INDEX["comp_rows"],
            top_k=5,
            min_sim=min_sim,
        )
        cybba_matches = _match_against(
            seg_name,
            _CATALOG_INDEX["cybba_vectorizer"],
            _CATALOG_INDEX["cybba_matrix"],
            _CATALOG_INDEX["cybba_rows"],
            top_k=5,
            min_sim=min_sim,
        )

        comp_max = max((m["similarity"] for m in comp_matches), default=0.0)
        cybba_max = max((m["similarity"] for m in cybba_matches), default=0.0)

        # Coverage gap score: competitors do this but Cybba doesn't
        gap_score = comp_max - cybba_max

        why_parts = []
        if comp_matches:
            top_comp = comp_matches[0]
            why_parts.append(
                f"Similar to '{top_comp['segment_name']}' by {top_comp['provider']} "
                f"({int(top_comp['similarity'] * 100)}% match)"
            )
        if cybba_max < 0.3:
            why_parts.append("low overlap with existing Cybba catalog — coverage gap")
        elif cybba_max < 0.6:
            why_parts.append("partial Cybba coverage — room to expand")

        suggestions.append({
            "id": str(uuid.uuid4()),
            "title": seg_name,
            "why": ". ".join(why_parts) if why_parts else "Potential new segment opportunity.",
            "proposed_l1": str(row.get("L1") or row.get("Taxonomy L1") or "").strip(),
            "proposed_l2": str(row.get("L2") or row.get("Taxonomy L2") or "").strip(),
            "seed_keywords": [],
            "seed_leaves": [],
            "competitor_matches": comp_matches,
            "cybba_matches": cybba_matches,
            "source": "web_assisted" if row.get("web_assisted") else "regular",
            "_gap_score": gap_score,
            "_comp_max": comp_max,
            "_cybba_max": cybba_max,
        })

    # Sort by gap_score descending (biggest coverage gap first)
    suggestions.sort(key=lambda x: x["_gap_score"], reverse=True)

    # Deduplicate by L1 to ensure diversity — keep best per L1, then fill remainder
    seen_l1: Dict[str, int] = {}
    diverse: List[Dict[str, Any]] = []

    for s in suggestions:
        l1 = s["proposed_l1"] or "Unknown"
        count = seen_l1.get(l1, 0)
        if count < 3:  # max 3 per L1 category
            seen_l1[l1] = count + 1
            diverse.append(s)
        if len(diverse) >= top_n:
            break

    # Fill remaining slots if needed
    used_ids = {s["id"] for s in diverse}
    for s in suggestions:
        if len(diverse) >= top_n:
            break
        if s["id"] not in used_ids:
            diverse.append(s)
            used_ids.add(s["id"])

    # Remove internal scoring fields before returning
    for s in diverse:
        s.pop("_gap_score", None)
        s.pop("_comp_max", None)
        s.pop("_cybba_max", None)

    return diverse[:top_n]


# ---------------------------------------------------------------------------
# Ollama analysis
# ---------------------------------------------------------------------------

def _build_analysis_prompt(
    segment_name: str,
    competitor_matches: List[Dict[str, Any]],
    cybba_matches: List[Dict[str, Any]],
    l1: str,
    l2: str,
) -> str:
    comp_lines = "\n".join(
        f"  - {m['segment_name']} ({m['provider']}, {int(m['similarity']*100)}% match)"
        for m in competitor_matches[:5]
    ) or "  (none found)"

    cybba_lines = "\n".join(
        f"  - {m['segment_name']} ({int(m['similarity']*100)}% match)"
        for m in cybba_matches[:5]
    ) or "  (none in Cybba catalog)"

    taxonomy = f"{l1} > {l2}".strip(" >") if (l1 or l2) else "Unknown"

    return f"""You are an expert in digital advertising audience segments and data marketplaces.

SEGMENT TO ANALYZE: "{segment_name}"
Taxonomy: {taxonomy}

COMPETITOR CATALOG MATCHES (segments that are similar in the competitor market):
{comp_lines}

CYBBA CATALOG MATCHES (existing Cybba segments that are similar):
{cybba_lines}

Based on the above, provide a concise analysis in JSON format with exactly these fields:
- "is_useful": true if this segment would be valuable to add to Cybba's catalog, false otherwise
- "helpfulness_reasoning": 1-2 sentences on why this segment is or isn't useful for Cybba
- "scaling_tips": 1-2 sentences on how this segment could help scale Cybba's audience offerings
- "competitor_context": 1-2 sentences on what competitors are doing with similar segments

Output ONLY valid JSON, no markdown, no explanation:
{{
  "is_useful": true,
  "helpfulness_reasoning": "...",
  "scaling_tips": "...",
  "competitor_context": "..."
}}"""


def _analyze_with_ollama(
    segment_name: str,
    competitor_matches: List[Dict[str, Any]],
    cybba_matches: List[Dict[str, Any]],
    l1: str,
    l2: str,
    *,
    ollama_url: str,
    model: str = "llama3.1",
    timeout: int = 60,
) -> Dict[str, Any]:
    prompt = _build_analysis_prompt(segment_name, competitor_matches, cybba_matches, l1, l2)
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 400},
        }
        resp = requests.post(ollama_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = str(resp.json().get("response", "")).strip()

        # Extract JSON from response
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return {
                "is_useful": bool(parsed.get("is_useful", True)),
                "helpfulness_reasoning": str(parsed.get("helpfulness_reasoning", "")),
                "scaling_tips": str(parsed.get("scaling_tips", "")),
                "competitor_context": str(parsed.get("competitor_context", "")),
            }
        # Fallback: try parsing entire raw response
        parsed = json.loads(raw)
        return {
            "is_useful": bool(parsed.get("is_useful", True)),
            "helpfulness_reasoning": str(parsed.get("helpfulness_reasoning", "")),
            "scaling_tips": str(parsed.get("scaling_tips", "")),
            "competitor_context": str(parsed.get("competitor_context", "")),
        }
    except Exception as e:
        logger.warning("Ollama analysis failed for '%s': %s", segment_name, e)
        return {
            "is_useful": None,
            "helpfulness_reasoning": "",
            "scaling_tips": "",
            "competitor_context": "",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@router.get("/api/suggestions/health")
def suggestions_health():
    return {"ok": True}


@router.post("/api/suggestions/generate")
def generate_suggestions(
    run_id: Optional[str] = Query(default=None),
    top_n: int = Query(default=25, ge=1, le=100),
) -> Dict[str, Any]:
    """
    Generate smart suggestions based on a completed pipeline run.
    Finds coverage gaps by matching generated segments against competitor + Cybba catalogs.
    """
    sid = str(uuid.uuid4())
    created_at = int(time.time())

    run_rows: List[Dict[str, Any]] = []

    # Try to load rows from the run
    if run_id:
        try:
            from src.persist import get_run_payload  # noqa: PLC0415
            base_dir = _get_base_dir()
            db_path = base_dir / "data" / "runs.db"
            payload = get_run_payload(db_path, run_id=run_id)
            if payload:
                run_rows = payload.get("rows") or []
                logger.info("Suggestions: loaded %d rows from run %s", len(run_rows), run_id)
        except Exception as e:
            logger.warning("Suggestions: could not load run %s: %s", run_id, e)

    if not run_rows:
        # No run data — return empty suggestions
        items = []
        payload_out = {
            "suggestion_set_id": sid,
            "created_at": created_at,
            "items": items,
            "note": "No run data found. Generate segments first, then open Suggestions.",
        }
        _SUGGESTIONS[sid] = payload_out
        return payload_out

    # Generate real suggestions from run rows
    try:
        items = _generate_suggestions_from_rows(run_rows, top_n=top_n)
        logger.info("Suggestions: generated %d items for run %s", len(items), run_id)
    except Exception as _e:
        logger.error("Suggestions: generation failed: %s", _e, exc_info=True)
        items = []

    payload_out = {
        "suggestion_set_id": sid,
        "created_at": created_at,
        "items": items,
    }
    _SUGGESTIONS[sid] = payload_out
    return payload_out


@router.post("/api/suggestions/analyze")
def analyze_suggestion(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask Ollama for an AI analysis of a specific segment.
    Body: {segment_name, competitor_matches, cybba_matches, l1?, l2?}
    """
    segment_name = str(body.get("segment_name") or "").strip()
    if not segment_name:
        return {"is_useful": None, "error": "segment_name is required"}

    competitor_matches = body.get("competitor_matches") or []
    cybba_matches = body.get("cybba_matches") or []
    l1 = str(body.get("l1") or "").strip()
    l2 = str(body.get("l2") or "").strip()

    try:
        base_dir = _get_base_dir()
        cfg = load_config(base_dir)
        sug_cfg = cfg.get("suggestions", {}) or {}
        ollama_url = str(sug_cfg.get("ollama_url", "http://host.docker.internal:11434/api/generate"))
        model = str(sug_cfg.get("model", "llama3.1"))
        timeout = int(sug_cfg.get("analysis_timeout_seconds", 60))
    except Exception:
        ollama_url = "http://host.docker.internal:11434/api/generate"
        model = "llama3.1"
        timeout = 60

    result = _analyze_with_ollama(
        segment_name,
        competitor_matches,
        cybba_matches,
        l1,
        l2,
        ollama_url=ollama_url,
        model=model,
        timeout=timeout,
    )
    return result


@router.post("/api/suggestions/select")
def select_suggestions(
    suggestion_set_id: str,
    selected_ids: List[str],
) -> Dict[str, Any]:
    if suggestion_set_id not in _SUGGESTIONS:
        return {"ok": False, "error": "suggestion_set_id not found"}

    _SUGGESTIONS["__selected__"] = {
        "suggestion_set_id": suggestion_set_id,
        "selected_ids": selected_ids,
        "updated_at": int(time.time()),
    }
    return {"ok": True, "selected_ids": selected_ids}


@router.get("/api/suggestions/selected")
def get_selected() -> Dict[str, Any]:
    return _SUGGESTIONS.get("__selected__", {"suggestion_set_id": None, "selected_ids": [], "updated_at": None})
