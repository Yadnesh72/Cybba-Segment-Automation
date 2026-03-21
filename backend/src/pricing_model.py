# backend/src/pricing_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None


PRICE_COLS = [
    "Digital Ad Targeting Price (CPM)",
    "Content Marketing Price (CPM)",
    "TV Targeting Price (CPM)",
    "Cost Per Click",
    "Programmatic % of Media",
    "CPM Cap",
    "Advertiser Direct % of Media",
]

CPM_COLS = {
    "Digital Ad Targeting Price (CPM)",
    "Content Marketing Price (CPM)",
    "TV Targeting Price (CPM)",
    "CPM Cap",
}
CPC_COLS = {"Cost Per Click"}
PCT_COLS = {"Programmatic % of Media", "Advertiser Direct % of Media"}


@dataclass
class PricingDefaults:
    provider_name: str = "Cybba"
    country: str = "USA"
    currency: str = "USD"


def _split_taxonomy(name: str) -> Tuple[str, str, str]:
    parts = [p.strip() for p in str(name or "").split(">") if p.strip()]
    if parts and parts[0].lower() == "cybba":
        parts = parts[1:]
    aud = parts[0] if len(parts) >= 1 else "UNKNOWN"
    l1 = parts[1] if len(parts) >= 2 else "UNKNOWN"
    l2 = parts[2] if len(parts) >= 3 else "UNKNOWN"
    return aud or "UNKNOWN", l1 or "UNKNOWN", l2 or "UNKNOWN"


def load_pricing_model(model_path: Path):
    if joblib is None:
        raise RuntimeError("joblib not installed. Add joblib + scikit-learn to requirements.")
    if not model_path.exists():
        raise FileNotFoundError(f"Pricing model not found at: {model_path}")
    return joblib.load(model_path)


def _round_outputs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in CPM_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").clip(lower=0.0).round(2)

    for c in CPC_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").clip(lower=0.0).round(2)

    for c in PCT_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").clip(lower=0.0, upper=100.0).round(0)

    return out


def _clip_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(lower=float(lo), upper=float(hi))


def _squash_to_range(x: float, lo: float, hi: float, temp_mult: float = 3.0) -> float:
    """
    Softly squash x into [lo, hi] using tanh.
    Unlike hard clip, this preserves variation near the bounds.

    temp_mult:
      higher => gentler squashing (more variation retained)
      lower  => stronger squashing (closer to hard clip)
    """
    if not (np.isfinite(x) and np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return x

    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    if half <= 0:
        return float(np.clip(x, lo, hi))

    t = half * float(temp_mult)
    return float(mid + half * np.tanh((x - mid) / (t + 1e-12)))


def _apply_calibration(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-column linear calibration learned from Cybba pricing CSV:
        calibrated = a + b * raw

    No-op if calibration_params not present.
    """
    params = getattr(model, "calibration_params", None)
    if not isinstance(params, dict) or not params:
        return df

    out = df.copy()
    for c in PRICE_COLS:
        v = params.get(c)
        if not v or c not in out.columns:
            continue
        try:
            a, b = v
            a = float(a)
            b = float(b)
        except Exception:
            continue
        out[c] = a + b * pd.to_numeric(out[c], errors="coerce")
    return out


def _apply_catalog_guardrails(
    model: Any,
    df: pd.DataFrame,
    audience: pd.Series,
    tax_l1: pd.Series,
    *,
    widen: float = 0.10,
    hard_clip: Optional[Dict[str, Tuple[float, float]]] = None,
    # ✅ NEW: softness control (higher keeps more spread)
    squash_temp_mult: float = 3.0,
) -> pd.DataFrame:
    """
    Catalog-like guardrails (data-driven), but with SOFT clipping to avoid "all values == p95".

    - Prefer tier_quantiles[(aud, l1)][col] = (p05, p95)
    - Else global_quantiles[col] = (p05, p95)
    - Widen by +/- widen*(hi-lo)
    - Then squash into [lo2, hi2] (tanh), instead of hard clipping

    If model wasn't trained with quantiles, this becomes a no-op (except optional hard_clip).
    """
    out = df.copy()

    tier_q = getattr(model, "tier_quantiles", None)
    glob_q = getattr(model, "global_quantiles", None)

    has_tier = isinstance(tier_q, dict)
    has_glob = isinstance(glob_q, dict)

    if not has_tier and not has_glob:
        if hard_clip:
            for c, (lo, hi) in hard_clip.items():
                if c in out.columns:
                    out[c] = _clip_series(out[c], lo, hi)
        return out

    aud = audience.reindex(out.index)
    l1 = tax_l1.reindex(out.index)

    for i in out.index:
        key = (str(aud.loc[i]), str(l1.loc[i]))

        for c in PRICE_COLS:
            lo_hi: Optional[Tuple[float, float]] = None

            if has_tier:
                row = tier_q.get(key)
                if isinstance(row, dict):
                    v = row.get(c)
                    if isinstance(v, (tuple, list)) and len(v) == 2:
                        lo_hi = (float(v[0]), float(v[1]))

            if lo_hi is None and has_glob:
                v2 = glob_q.get(c)
                if isinstance(v2, (tuple, list)) and len(v2) == 2:
                    lo_hi = (float(v2[0]), float(v2[1]))

            if lo_hi is None:
                continue

            lo, hi = lo_hi
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                continue

            pad = float(widen) * (hi - lo)
            lo2 = lo - pad
            hi2 = hi + pad

            x = float(pd.to_numeric(pd.Series([out.loc[i, c]]), errors="coerce").iloc[0])
            if not np.isfinite(x):
                continue

            out.loc[i, c] = _squash_to_range(x, lo2, hi2, temp_mult=float(squash_temp_mult))

    # absolute safety net if you want
    if hard_clip:
        for c, (lo, hi) in hard_clip.items():
            if c in out.columns:
                out[c] = _clip_series(out[c], lo, hi)

    return out


def _baseline_for_rows(model: Any, audience: pd.Series, tax_l1: pd.Series) -> pd.DataFrame:
    """
    Tier baseline by (Audience Type, Taxonomy L1).
    """
    baseline_table = getattr(model, "baseline_table", None)
    fallback_means = getattr(model, "fallback_means", {}) or {}

    base = pd.DataFrame(index=audience.index, columns=PRICE_COLS, dtype=float)

    for i in base.index:
        key = (str(audience.loc[i]), str(tax_l1.loc[i]))
        row = None
        if isinstance(baseline_table, dict):
            row = baseline_table.get(key)

        if isinstance(row, dict):
            for c in PRICE_COLS:
                v = row.get(c)
                if v is not None and pd.notna(v):
                    base.loc[i, c] = float(v)
                else:
                    base.loc[i, c] = float(fallback_means.get(c, 0.0))
        else:
            for c in PRICE_COLS:
                base.loc[i, c] = float(fallback_means.get(c, 0.0))

    return base


def _make_query_text(full_name: pd.Series, desc: pd.Series, comps: pd.Series) -> pd.Series:
    return (
        full_name.fillna("").astype(str)
        + " | "
        + desc.fillna("").astype(str)
        + " | "
        + comps.fillna("").astype(str)
    ).str.strip()


def _knn_predict_prices(
    model: Any,
    audience: pd.Series,
    tax_l1: pd.Series,
    query_text: pd.Series,
    k: int = 50,
    min_sim: float = 0.15,
    *,
    return_top_sim: bool = False,
):
    """
    kNN weighted pricing:
    - Vectorize query text
    - Restrict neighbors to same (audience, l1) if possible
    - Weighted average by cosine similarity
    """
    vec = getattr(model, "knn_vectorizer", None)
    X_ref = getattr(model, "knn_matrix", None)
    ref_prices = getattr(model, "knn_prices", None)
    ref_aud = getattr(model, "knn_audience", None)
    ref_l1 = getattr(model, "knn_tax_l1", None)
    fallback_means = getattr(model, "fallback_means", {}) or {}

    out = pd.DataFrame(index=query_text.index, columns=PRICE_COLS, dtype=float)
    # Track the strongest neighbor similarity per query so we can adapt blending.
    top_sim = pd.Series(np.zeros(len(query_text), dtype=float), index=query_text.index)

    if vec is None or X_ref is None or ref_prices is None or ref_aud is None or ref_l1 is None:
        for c in PRICE_COLS:
            out[c] = float(fallback_means.get(c, 0.0))
        if return_top_sim:
            return out, top_sim
        return out

    # Vectorize queries
    Xq = vec.transform(query_text.tolist())  # sparse

    # Build tier->indices once
    tier_to_idx: Dict[Tuple[str, str], List[int]] = {}
    for idx, (a, l) in enumerate(zip(ref_aud, ref_l1)):
        key = (str(a), str(l))
        tier_to_idx.setdefault(key, []).append(idx)

    tier_to_idx_np: Dict[Tuple[str, str], np.ndarray] = {
        k2: np.asarray(v2, dtype=int) for k2, v2 in tier_to_idx.items()
    }

    for i in range(Xq.shape[0]):
        key = (str(audience.iloc[i]), str(tax_l1.iloc[i]))
        cand = tier_to_idx_np.get(key)

        # if tier too small, do global neighbors
        use_global = cand is None or len(cand) < 200

        if use_global:
            sims = (Xq[i] @ X_ref.T).toarray().reshape(-1)
            ref_idx = None
        else:
            sims = (Xq[i] @ X_ref[cand].T).toarray().reshape(-1)
            ref_idx = cand

        if sims.size == 0:
            for c in PRICE_COLS:
                out.iloc[i, out.columns.get_loc(c)] = float(fallback_means.get(c, 0.0))
            top_sim.iloc[i] = 0.0
            continue

        kk = min(int(k), int(len(sims)))
        if kk <= 0:
            for c in PRICE_COLS:
                out.iloc[i, out.columns.get_loc(c)] = float(fallback_means.get(c, 0.0))
            top_sim.iloc[i] = 0.0
            continue

        top = np.argpartition(-sims, kth=kk - 1)[:kk]
        top_sims = sims[top]

        # Record the strongest similarity among the candidate neighbors (before min_sim filtering).
        try:
            top_sim.iloc[i] = float(np.max(top_sims)) if top_sims.size else 0.0
        except Exception:
            top_sim.iloc[i] = 0.0

        keep = top_sims >= float(min_sim)
        top = top[keep]
        top_sims = top_sims[keep]

        if len(top) == 0:
            for c in PRICE_COLS:
                out.iloc[i, out.columns.get_loc(c)] = float(fallback_means.get(c, 0.0))
            top_sim.iloc[i] = 0.0
            continue

        # map to global indices if tiered
        if ref_idx is not None:
            top_global = ref_idx[top]
        else:
            top_global = top

        nbr_prices = ref_prices.iloc[top_global]

        w = top_sims.astype(float)
        w = w / (w.sum() + 1e-12)

        for c in PRICE_COLS:
            colv = pd.to_numeric(nbr_prices[c], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(colv)
            if not m.any():
                out.iloc[i, out.columns.get_loc(c)] = float(fallback_means.get(c, 0.0))
                continue
            ww = w[m]
            ww = ww / (ww.sum() + 1e-12)
            out.iloc[i, out.columns.get_loc(c)] = float(np.sum(colv[m] * ww))

    if return_top_sim:
        return out, top_sim
    return out


@dataclass
class PricingModelBundle:
    price_cols: List[str]
    fallback_means: Dict[str, float]
    baseline_table: Dict[Tuple[str, str], Dict[str, float]]

    tier_quantiles: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]
    global_quantiles: Dict[str, Tuple[float, float]]

    calibration_params: Dict[str, Tuple[float, float]]

    knn_vectorizer: Any
    knn_matrix: Any
    knn_prices: pd.DataFrame
    knn_audience: List[str]
    knn_tax_l1: List[str]


def predict_prices(
    model: Any,
    segments_df: pd.DataFrame,
    defaults: Optional[PricingDefaults] = None,
    *,
    k: int = 40,
    min_sim: float = 0.12,
    blend_baseline: float = 0.35,
    quantile_widen: float = 0.12,
    hard_clip: Optional[Dict[str, Tuple[float, float]]] = None,
    # ✅ NEW: controls how "hard" quantile guardrails feel (higher => more variation)
    squash_temp_mult: float = 4.0,
    # ✅ NEW: adaptive blending controls
    baseline_min: float = 0.10,
    baseline_gamma: float = 1.5,
) -> pd.DataFrame:
    """
    1) kNN pricing
    2) blend with tier baseline
    3) soft-guardrail to catalog quantiles (tier->global) + optional hard_clip
    4) round

    NOTE: Calibration is currently disabled (commented), per your request.
    """
    defaults = defaults or PricingDefaults()

    # segment name column
    if "Proposed New Segment Name" in segments_df.columns:
        name_col = "Proposed New Segment Name"
    elif "New Segment Name" in segments_df.columns:
        name_col = "New Segment Name"
    else:
        raise ValueError("predict_prices expected 'Proposed New Segment Name' or 'New Segment Name'.")

    desc_col = "Segment Description" if "Segment Description" in segments_df.columns else None
    comps_col = "Non Derived Segments utilized" if "Non Derived Segments utilized" in segments_df.columns else None

    full_name = segments_df[name_col].fillna("").astype(str)
    seg_desc = (
        segments_df[desc_col].fillna("").astype(str)
        if desc_col
        else pd.Series([""] * len(segments_df), index=segments_df.index)
    )
    seg_comps = (
        segments_df[comps_col].fillna("").astype(str)
        if comps_col
        else pd.Series([""] * len(segments_df), index=segments_df.index)
    )

    aud_l1_l2 = full_name.map(_split_taxonomy)
    audience = aud_l1_l2.map(lambda t: t[0])
    tax_l1 = aud_l1_l2.map(lambda t: t[1])

    baseline_df = _baseline_for_rows(model, audience, tax_l1)

    query_text = _make_query_text(full_name, seg_desc, seg_comps)
    knn_df, top_sim = _knn_predict_prices(
        model,
        audience,
        tax_l1,
        query_text,
        k=int(k),
        min_sim=float(min_sim),
        return_top_sim=True,
    )

    # Blend (baseline stabilizes), but adapt by confidence:
    # - If we have strong nearest-neighbor similarity, rely more on kNN (more natural variation).
    # - If similarity is weak / cold-start, lean more on the baseline (stability).
    b = float(blend_baseline)
    b = min(max(b, 0.0), 1.0)

    b_min = float(baseline_min)
    b_min = min(max(b_min, 0.0), 1.0)

    gamma = float(baseline_gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        gamma = 1.0

    # Clamp similarities into [0, 1] and compute per-row baseline weight.
    sim = pd.to_numeric(top_sim, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)

    # Higher sim => smaller baseline weight, but never below b_min.
    b_row = b_min + (b - b_min) * ((1.0 - sim) ** gamma)
    b_row = b_row.clip(lower=0.0, upper=1.0)

    # Vectorized row-wise blend.
    out = knn_df.astype(float).mul(1.0 - b_row, axis=0) + baseline_df.astype(float).mul(b_row, axis=0)

    # calibration disabled for now (you already tested)
    # out = _apply_calibration(model, out)

    out = _apply_catalog_guardrails(
        model,
        out,
        audience=audience,
        tax_l1=tax_l1,
        widen=float(quantile_widen),
        hard_clip=hard_clip,
        squash_temp_mult=float(squash_temp_mult),
    )

    out = _round_outputs(out)
    return out[PRICE_COLS]