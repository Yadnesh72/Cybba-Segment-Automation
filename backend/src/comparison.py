from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.pipeline import load_config

router = APIRouter()

_INDEX: Dict[str, Any] = {
    "ready": False,
    "vectorizer": None,
    "matrix": None,
    "rows": None,
    "texts": None,
}

def _normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split())

def _get_base_dir() -> Path:
    # backend/src/comparison.py -> backend/
    return Path(__file__).resolve().parents[1]

def _build_index(base_dir: Path) -> None:
    cfg = load_config(base_dir)

    input_dir = Path(cfg["paths"]["input_dir"])
    input_a = input_dir / cfg["files"]["input_a"]
    if not input_a.exists():
        raise HTTPException(status_code=404, detail=f"Catalog not found: {input_a}")

    df = pd.read_csv(input_a, dtype=str, keep_default_na=False)

    provider_col = "Provider Name"
    name_col = "Segment Name"
    desc_col = "Segment Description" if "Segment Description" in df.columns else None

    if provider_col not in df.columns or name_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Catalog missing required cols: {provider_col}, {name_col}",
        )

    cybba_name = str(cfg.get("provider", {}).get("cybba_name", "Cybba")).strip().lower()

    comp = df[df[provider_col].astype(str).str.strip().str.lower() != cybba_name].copy()

    texts: List[str] = []
    rows: List[Dict[str, Any]] = []

    for _, r in comp.iterrows():
        provider = str(r.get(provider_col, "")).strip() or "Unknown"
        seg = str(r.get(name_col, "")).strip()
        if not seg:
            continue

        desc = str(r.get(desc_col, "")).strip() if desc_col else ""
        text = seg if not desc else f"{seg} | {desc}"

        rows.append(
            {
                "competitor": provider,
                "segment_name": seg,
                "description": desc or None,
            }
        )
        texts.append(_normalize(text))

    if not texts:
        raise HTTPException(status_code=400, detail="No competitor segments found to index.")

    # safer defaults for robustness
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
        max_features=50000,
    )

    try:
        mat = vec.fit_transform(texts)
    except ValueError:
        # fallback if something goes wrong due to sparsity
        vec = TfidfVectorizer(ngram_range=(1, 1), stop_words="english", min_df=1, max_features=50000)
        mat = vec.fit_transform(texts)

    _INDEX.update(
        {
            "ready": True,
            "vectorizer": vec,
            "matrix": mat,
            "rows": rows,
            "texts": texts,
        }
    )

def _ensure_index(base_dir: Path) -> None:
    if _INDEX["ready"]:
        return
    _build_index(base_dir)

@router.get("/api/competitor_matches")
def competitor_matches(
    query: str = Query(..., description="Your segment name"),
    top_k: int = Query(20, ge=1, le=50),
    pool: int = Query(200, ge=20, le=2000),
    min_sim: float = Query(0.0, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    base_dir = _get_base_dir()
    _ensure_index(base_dir)

    vec: TfidfVectorizer = _INDEX["vectorizer"]
    mat = _INDEX["matrix"]
    rows: List[Dict[str, Any]] = _INDEX["rows"]

    q = _normalize(query)
    qv = vec.transform([q])

    sims = cosine_similarity(qv, mat).ravel()
    valid = np.where(sims >= float(min_sim))[0]
    if valid.size == 0:
        return {"query": query, "matches": []}

    idxs = valid[np.argsort(-sims[valid])][:pool]
   

    # diversify by competitor: best per competitor first
    best_by_comp: Dict[str, Tuple[int, float]] = {}
    ordered: List[Tuple[int, float]] = []

    for i in idxs:
        r = rows[int(i)]
        comp = r["competitor"]
        s = float(sims[int(i)])
        if comp not in best_by_comp:
            best_by_comp[comp] = (int(i), s)

    per_comp = sorted(best_by_comp.values(), key=lambda x: x[1], reverse=True)
    ordered.extend(per_comp)

    used = {i for i, _ in ordered}
    for i in idxs:
        ii = int(i)
        if ii in used:
            continue
        ordered.append((ii, float(sims[ii])))
        if len(ordered) >= top_k:
            break

    out = []
    for ii, s in ordered[:top_k]:
        item = dict(rows[ii])
        item["score"] = round(s, 4)
        out.append(item)

    return {"query": query, "matches": out}