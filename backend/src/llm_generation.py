#!/usr/bin/env python3
# backend/src/llm_generation.py
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

OLLAMA_URL_DEFAULT = "http://localhost:11434/api/generate"

# Reuse HTTP connection (big speedup)
_SESSION = requests.Session()


def _clean(s: object) -> str:
    return str(s or "").strip()


def _stable_key(*parts: str) -> str:
    base = "|".join(_clean(p) for p in parts).encode("utf-8")
    return hashlib.sha256(base).hexdigest()


def _extract_json_any(text: str) -> Optional[Any]:
    """
    Extract first JSON object OR array from model response.
    Robust to stray text before/after JSON.
    """
    t = _clean(text)
    if not t:
        return None

    # direct parse
    try:
        return json.loads(t)
    except Exception:
        pass

    # try find an array first
    m = re.search(r"\[.*\]", t, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # then try object
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _build_prompt_batch(items: List[Dict[str, str]]) -> str:
    """
    Ask for a JSON ARRAY of objects with {id,leaf,l2,notes}.
    Smaller prompt than repeating full instructions N times.
    """
    payload: List[Dict[str, str]] = []
    for i, it in enumerate(items):
        payload.append(
            {
                "id": str(i),
                "competitor_name": it.get("competitor_name", ""),
                "competitor_desc": it.get("competitor_desc", ""),
                "components": it.get("components", ""),
                "draft_l1": it.get("draft_l1", ""),
                "draft_l2": it.get("draft_l2", ""),
                "draft_leaf": it.get("draft_leaf", ""),
            }
        )

    return f"""
You refine Cybba audience segment taxonomy.
Return ONLY valid JSON (no markdown, no prose).

You will receive an array of inputs. For each input, return an output object.

Hard rules:
- Output MUST be a JSON array. Each element MUST have: "id", "leaf", "l2", "notes"
- "leaf" is required (string). "l2" may be "" if unchanged/unknown. "notes" may be "".
- Leaf must be advertiser-friendly and generic:
  - no brand names, no specific companies, no universities
  - concise: 2–6 words before suffix
  - suffix only if appropriate (use at most one): Decision Makers, Buyers, Enthusiasts, Shoppers, Fans, Users, Professionals
  - do NOT repeat suffix twice
- Do NOT include "Cybba" anywhere
- Do NOT output ABM, Top Companies, or entity lists
- L2 should be short and not duplicate L1 (if unsure, return "")

INPUTS (JSON):
{json.dumps(payload, ensure_ascii=False)}

Return JSON only.
""".strip()


# =============================================================================
# FAST PATH (Option B): SINGLE REQUEST BATCH
# =============================================================================
def refine_segments_taxonomy_batch(
    *,
    batch: List[Dict[str, str]],
    model: str = "llama3.1",
    ollama_url: str = OLLAMA_URL_DEFAULT,
    timeout: int = 45,
    cache: Optional[Dict[str, dict]] = None,
    num_predict: int = 90,  # lower = faster
) -> List[Dict[str, Any]]:
    """
    Batch refine using ONE Ollama call.

    Input: batch of dicts with keys:
      competitor_name, competitor_desc, components, draft_l1, draft_l2, draft_leaf

    Output: list aligned to input order:
      {leaf, l2, notes, ok}

    Never raises.
    """
    if cache is None:
        cache = {}

    if not batch:
        return []

    # Prepare result array
    results: List[Dict[str, Any]] = [{"leaf": "", "l2": "", "notes": "", "ok": False} for _ in batch]

    # Fill from cache; collect uncached
    uncached_items: List[Dict[str, str]] = []
    uncached_map: List[Tuple[int, str]] = []  # (orig_idx, cache_key)

    for idx, it in enumerate(batch):
        cname = _clean(it.get("competitor_name", ""))
        cdesc = _clean(it.get("competitor_desc", ""))
        comps = _clean(it.get("components", ""))
        l1 = _clean(it.get("draft_l1", ""))
        l2 = _clean(it.get("draft_l2", ""))
        leaf = _clean(it.get("draft_leaf", ""))

        if not cname:
            results[idx] = {"leaf": "", "l2": "", "notes": "missing competitor_name", "ok": False}
            continue

        key = _stable_key(cname, cdesc, comps, l1, l2, leaf, model)
        if key in cache:
            out = cache[key]
            # cache stores leaf/l2/notes only
            results[idx] = {**out, "ok": True}
            continue

        # keep prompt compact
        uncached_items.append(
            {
                "competitor_name": cname,
                "competitor_desc": cdesc[:220],
                "components": comps[:260],
                "draft_l1": l1,
                "draft_l2": l2,
                "draft_leaf": leaf,
            }
        )
        uncached_map.append((idx, key))

    # All cached
    if not uncached_items:
        return results

    prompt = _build_prompt_batch(uncached_items)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.15,
            "top_p": 0.9,
            "num_predict": int(num_predict),
            # Optional tuning knobs:
            # "num_ctx": 2048,
            # "num_thread": 8,
        },
    }

    try:
        resp = _SESSION.post(ollama_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        raw = _clean(data.get("response", ""))

        parsed = _extract_json_any(raw)
        if not isinstance(parsed, list):
            for (orig_idx, _) in uncached_map:
                results[orig_idx] = {"leaf": "", "l2": "", "notes": "json_parse_failed", "ok": False}
            return results

        # parsed is list of {id, leaf, l2, notes}
        by_id: Dict[int, dict] = {}
        for obj in parsed:
            if not isinstance(obj, dict):
                continue
            try:
                i = int(obj.get("id", -1))
            except Exception:
                continue
            by_id[i] = obj

        for local_i, (orig_idx, cache_key) in enumerate(uncached_map):
            obj = by_id.get(local_i)
            if not obj:
                results[orig_idx] = {"leaf": "", "l2": "", "notes": "missing_item_in_response", "ok": False}
                continue

            out = {
                "leaf": _clean(obj.get("leaf", "")),
                "l2": _clean(obj.get("l2", "")),
                "notes": _clean(obj.get("notes", "")),
            }

            # leaf is required
            if not out["leaf"]:
                results[orig_idx] = {"leaf": "", "l2": "", "notes": "empty_leaf", "ok": False}
                continue

            cache[cache_key] = out
            results[orig_idx] = {**out, "ok": True}

        return results

    except Exception as e:
        msg = f"ollama_failed: {type(e).__name__}"
        for (orig_idx, _) in uncached_map:
            results[orig_idx] = {"leaf": "", "l2": "", "notes": msg, "ok": False}
        return results


# =============================================================================
# SINGLE ITEM (Backward compatible wrapper)
# =============================================================================
def refine_segment_taxonomy(
    *,
    competitor_name: str,
    competitor_desc: str,
    components: str,
    draft_l1: str,
    draft_l2: str,
    draft_leaf: str,
    model: str = "llama3.1",
    ollama_url: str = OLLAMA_URL_DEFAULT,
    timeout: int = 45,
    cache: Optional[Dict[str, dict]] = None,
) -> Dict[str, Any]:
    """
    Old single-item call signature preserved.
    Uses the fast batch endpoint with a 1-item batch.
    """
    return refine_segments_taxonomy_batch(
        batch=[
            {
                "competitor_name": competitor_name,
                "competitor_desc": competitor_desc,
                "components": components,
                "draft_l1": draft_l1,
                "draft_l2": draft_l2,
                "draft_leaf": draft_leaf,
            }
        ],
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
        cache=cache,
        num_predict=140,
    )[0]


# =============================================================================
# OPTIONAL FALLBACK: PARALLEL SINGLE-ITEM CALLS
# Use only if your model sometimes fails to return a clean JSON array.
# =============================================================================
def refine_segments_taxonomy_parallel(
    *,
    items: List[Dict[str, str]],
    model: str = "llama3.1",
    ollama_url: str = OLLAMA_URL_DEFAULT,
    timeout: int = 45,
    max_workers: int = 6,
    cache: Optional[Dict[str, dict]] = None,
) -> List[Dict[str, Any]]:
    """
    Parallel wrapper around refine_segment_taxonomy().

    items: list of dicts with keys:
      competitor_name, competitor_desc, components, draft_l1, draft_l2, draft_leaf

    Returns same length/order as items:
      {leaf, l2, notes, ok}
    Never raises.
    """
    if cache is None:
        cache = {}

    if not items:
        return []

    results: List[Optional[Dict[str, Any]]] = [None] * len(items)

    def _run_one(i: int, it: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
        out = refine_segment_taxonomy(
            competitor_name=it.get("competitor_name", ""),
            competitor_desc=it.get("competitor_desc", ""),
            components=it.get("components", ""),
            draft_l1=it.get("draft_l1", ""),
            draft_l2=it.get("draft_l2", ""),
            draft_leaf=it.get("draft_leaf", ""),
            model=model,
            ollama_url=ollama_url,
            timeout=timeout,
            cache=cache,
        )
        return i, out

    workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_one, i, it) for i, it in enumerate(items)]
        for fut in as_completed(futs):
            try:
                i, out = fut.result()
                results[i] = out
            except Exception:
                pass

    for i in range(len(results)):
        if results[i] is None:
            results[i] = {"leaf": "", "l2": "", "notes": "batch_worker_failed", "ok": False}

    return results