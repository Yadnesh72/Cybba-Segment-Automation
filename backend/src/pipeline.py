# backend/src/pipeline.py
from __future__ import annotations

import io
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import time
import threading
import queue

import pandas as pd
import yaml

from src.llm_descriptions import generate_descriptions, generate_segment_description
from src.segment_expansion_model import generate_proposals
from src.validators import validate_and_prepare

# Pricing model (trained .joblib inference)
from src.pricing_model import PRICE_COLS, PricingDefaults, load_pricing_model, predict_prices

# ----------------------------
# Logging (prints may be buffered; logging is safer)
# ----------------------------
# ----------------------------
# Logging (minimal, reload-safe)
# ----------------------------
logger = logging.getLogger("pipeline")
tax_log = logging.getLogger("taxonomy")
price_log = logging.getLogger("pricing")
val_log = logging.getLogger("validators")
gen_log = logging.getLogger("generator")

def _ensure_minimal_logging() -> None:
    """
    Minimal, readable, avoids duplicate handlers (uvicorn --reload).
    Uses LOG_LEVEL env var (default INFO).
    """
    import os

    level_name = (os.environ.get("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # avoid handler duplication on reload
    if root.handlers:
        return

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

_ensure_minimal_logging()

# Shim
import sys
from src.taxonomy_bundle import TaxonomyL1Bundle

sys.modules.get("__mp_main__", sys).TaxonomyL1Bundle = TaxonomyL1Bundle
sys.modules.get("__main__", sys).TaxonomyL1Bundle = TaxonomyL1Bundle

# ----------------------------
# In-memory cache for LLM descriptions (persists while FastAPI process is running)
# ----------------------------
DESCRIPTION_CACHE: Dict[str, str] = {}

# ----------------------------
# Config
# ----------------------------
def load_config(base_dir: Path) -> dict:
    cfg_path = base_dir / "config" / "config.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


# ----------------------------
# Normalization helpers (for UI metrics)
# ----------------------------
def clean(s: Any) -> str:
    return str(s or "").strip()


def normalize_name(s: Any) -> str:
    """
    Must match the intent of validators.normalize_name:
    - lowercase
    - remove punctuation
    - collapse whitespace
    """
    t = clean(s).lower()
    t = re.sub(r"[^\w\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def safe_quantiles(series: pd.Series) -> Dict[str, float]:
    """
    Returns quantiles on numeric series, ignoring NaNs.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "min": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "mean": float("nan"),
        }
    return {
        "min": float(s.min()),
        "p10": float(s.quantile(0.10)),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "mean": float(s.mean()),
    }

# Time
class _Timer:
    def __init__(self, log, name):
        self.log, self.name = log, name
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.log.info("[%s] %.2fs", self.name, time.perf_counter() - self.t0)


def _df_to_rows(df: pd.DataFrame) -> List[dict]:
    """
    Convert DataFrame to JSON-safe rows (avoid NaN/NA).
    """
    return df.replace({pd.NA: None}).where(pd.notnull(df), None).to_dict(orient="records")

def _log_pipeline_start(cfg: dict, *, max_rows: Optional[int], allowed_categories: Optional[List[str]]) -> None:
    use_gate = bool((cfg.get("generation", {}) or {}).get("use_allowed_categories_gating", False))
    logger.info(
        "[PIPE] start max_rows=%s gate=%s allowed_categories=%d",
        max_rows if isinstance(max_rows, int) else "None",
        use_gate,
        len(allowed_categories or []),
    )

def _log_pipeline_summary(summary: Dict[str, Any]) -> None:
    # Keep this compact and consistent
    logger.info(
        "[PIPE] proposals=%s validated_total=%s output=%s covered=%s "
        "drop(naming=%s underived=%s net_new=%s) collisions=%s "
        "taxonomy=%s pricing=%s",
        summary.get("total_proposals"),
        summary.get("validated_total"),
        summary.get("validated_generated"),
        summary.get("covered"),
        summary.get("dropped_naming"),
        summary.get("dropped_underived_only"),
        summary.get("dropped_net_new"),
        summary.get("collisions_resolved"),
        "on" if summary.get("taxonomy_enabled") else "off",
        "on" if summary.get("pricing_enabled") else "off",
    )
# ----------------------------
# Output formatting (template-driven)
# ----------------------------
def load_output_template(cfg: dict) -> pd.DataFrame:
    """
    Reads Output_format.csv so we match its column order exactly.
    config.yml:
      output:
        template_file: "Output_format.csv"
    """
    template_file = cfg.get("output", {}).get("template_file", "")
    if not template_file:
        cols = [
            "New Segment Name",
            "Non Derived Segments utilized",
            "Segment Description",
            "Digital Ad Targeting Price (CPM)",
            "Content Marketing Price (CPM)",
            "TV Targeting Price (CPM)",
            "Cost Per Click",
            "Programmatic % of Media",
            "CPM Cap",
            "Advertiser Direct % of Media",
        ]
        return pd.DataFrame(columns=cols)

    input_dir = Path(cfg["paths"]["input_dir"])
    tpath = Path(template_file)
    if not tpath.is_absolute():
        tpath = input_dir / template_file

    if not tpath.exists():
        raise FileNotFoundError(f"Output template not found: {tpath}")

    tdf = pd.read_csv(tpath, dtype=str, keep_default_na=False)

    # even if empty, preserve header/columns
    return pd.DataFrame(columns=list(tdf.columns))


def build_final_from_template(validated_df: pd.DataFrame, template_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a dataframe whose columns match Output_format.csv exactly.
    """
    if template_df is None or len(template_df.columns) == 0:
        raise ValueError("Template dataframe has no columns.")

    out_cols = list(template_df.columns)
    out = pd.DataFrame(columns=out_cols)

    # Required mappings
    if "New Segment Name" in out_cols and "Proposed New Segment Name" in validated_df.columns:
        out["New Segment Name"] = validated_df["Proposed New Segment Name"].astype(str)

    if "Non Derived Segments utilized" in out_cols and "Non Derived Segments utilized" in validated_df.columns:
        out["Non Derived Segments utilized"] = validated_df["Non Derived Segments utilized"].astype(str)

    # Segment Description (supports LLM-generated descriptions)
    if "Segment Description" in out_cols:
        if "Segment Description" in validated_df.columns:
            out["Segment Description"] = validated_df["Segment Description"].astype(str)
        else:
            out["Segment Description"] = ""

    # Fill remaining columns (pricing values can be overwritten by model if enabled)
    for c in out_cols:
        if c in out.columns and len(out[c].dropna()) > 0:
            continue

        if c in validated_df.columns:
            out[c] = validated_df[c]
        else:
            # numeric-ish defaults for pricing fields; otherwise blank
            if "Price" in c or "CPM" in c or "%" in c or "Cost" in c:
                out[c] = "0"
            else:
                out[c] = ""

    return out[out_cols]


# ----------------------------
# Scoring
# ----------------------------
def add_uniqueness_and_rank(validated_df: pd.DataFrame) -> pd.DataFrame:
    """
    uniqueness_score = 1 - closest_cybba_similarity
    rank_score = 0.7 * composition_similarity + 0.3 * uniqueness_score
    """
    df = validated_df.copy()

    closest = pd.to_numeric(
        df.get("Closest Cybba Similarity", pd.Series([float("nan")] * len(df))),
        errors="coerce",
    )
    comp = pd.to_numeric(
        df.get("Composition Similarity", pd.Series([float("nan")] * len(df))),
        errors="coerce",
    )

    # uniqueness = 1 - closest (if NaN, treat as 1.0 unique)
    df["uniqueness_score"] = closest.apply(lambda v: 1.0 - float(v) if pd.notna(v) else 1.0)

    def rank_row(c: Any, u: Any) -> float:
        cc = float(c) if pd.notna(c) else 0.0
        uu = float(u) if pd.notna(u) else 0.0
        return 0.7 * cc + 0.3 * uu

    df["rank_score"] = [rank_row(c, u) for c, u in zip(comp, df["uniqueness_score"])]

    return df


def _sort_for_generation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make generation deterministic:
    Prefer rank_score desc, else Composition Similarity desc.
    """
    out = df.copy()
    if "rank_score" in out.columns:
        return out.sort_values("rank_score", ascending=False)
    if "Composition Similarity" in out.columns:
        return out.sort_values("Composition Similarity", ascending=False)
    return out


def _apply_cap(df: pd.DataFrame, max_rows: Optional[int]) -> pd.DataFrame:
    if isinstance(max_rows, int) and max_rows > 0:
        return df.head(max_rows).copy()
    return df


# ----------------------------
# ✅ Pricing (Phase 1: model inference)
# ----------------------------
def _pricing_config(cfg: dict) -> Tuple[bool, Optional[str], PricingDefaults]:
    pricing_cfg = cfg.get("pricing_model", {}) or {}
    enabled = bool(pricing_cfg.get("enable", False))

    model_path = pricing_cfg.get("model_path")
    defaults_cfg = pricing_cfg.get("defaults", {}) or {}
    defaults = PricingDefaults(
        provider_name=str(defaults_cfg.get("provider_name", "Cybba")),
        country=str(defaults_cfg.get("country", "USA")),
        currency=str(defaults_cfg.get("currency", "USD")),
    )
    return enabled, model_path, defaults


def _resolve_model_path(base_dir: Path, model_path: str) -> Path:
    """
    Resolve pricing model path safely:
    - expanduser (~)
    - if relative, resolve against base_dir
    """
    p = Path(model_path).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


# ✅ FIX: cache includes file mtime (and size) so overwriting the model triggers reload
@lru_cache(maxsize=16)
def _cached_pricing_model(model_path_abs: str, mtime: float, size: int):
    return load_pricing_model(Path(model_path_abs))


def _load_pricing_model_with_fingerprint(resolved: Path):
    st = resolved.stat()
    return _cached_pricing_model(str(resolved), float(st.st_mtime), int(st.st_size))


def _apply_pricing_if_enabled(
    base_dir: Path,
    cfg: dict,
    validated_scored: pd.DataFrame,
    final_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    If enabled in config.yml, loads trained model and fills pricing columns
    in final_df (template-driven output) based on validated_scored content.

    Returns (final_df, pricing_error_message_or_None).
    Never raises: we don't want to kill SSE streaming.
    """
    price_log.info("[PRICE] start")

    enabled, model_path, defaults = _pricing_config(cfg)
    if not enabled:
        price_log.info("[PRICE] disabled")
        return final_df, None

    if not model_path:
        msg = "pricing_model.enable=true but pricing_model.model_path is missing in config.yml"
        price_log.error("[PRICE] %s", msg)
        return final_df, msg

    if validated_scored is None or len(validated_scored) == 0:
        price_log.info("[PRICE] skip (no rows)")
        return final_df, None

    try:
        resolved = _resolve_model_path(base_dir, model_path)
        if not resolved.exists():
            msg = f"Pricing model not found at: {resolved}"
            logger.error("[pricing] %s", msg)
            return final_df, msg

        price_log.info("[PRICE] model=%s", resolved)

        # ✅ always reloads after retrain overwrite (mtime/size changes)
        model = _load_pricing_model_with_fingerprint(resolved)

        price_log.info("[PRICE] predict rows=%d", len(validated_scored))
        prices_df = predict_prices(model, validated_scored, defaults=defaults)

        # minimal sanity check
        try:
            nonzero = int((prices_df[PRICE_COLS].astype(float) != 0).any(axis=None))
        except Exception:
            nonzero = -1
        price_log.info("[PRICE] done nonzero_any=%s", nonzero if nonzero != -1 else "unknown")

        out = final_df.copy()
        for c in PRICE_COLS:
            if c in out.columns and c in prices_df.columns:
                out[c] = prices_df[c]

        return out, None

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        price_log.exception("[PRICE] failed: %s", msg)
        return final_df, msg


# ----------------------------
# ✅ Taxonomy (L1 classifier + L2 retriever)
# ----------------------------
def _taxonomy_config(cfg: dict) -> Tuple[bool, Optional[str], Optional[str], int, Optional[float]]:
    """
    config.yml expected:
      taxonomy_model:
        enable: true|false
        l1_model_path: "Data/Models/cybba_taxonomy_L1.joblib"
        l2_model_path: "Data/Models/cybba_taxonomy_L2.joblib"
        l2_top_k: 5
        l2_min_similarity: 0.0   # optional
        output_l1_col: "Predicted L1"  # optional rename
        output_l2_col: "Predicted L2"  # optional rename
        output_l1_conf_col: "Predicted L1 Confidence"  # optional rename
        output_l2_sim_col: "Predicted L2 Similarity"   # optional rename
    """
    tcfg = cfg.get("taxonomy_model", {}) or {}
    enabled = bool(tcfg.get("enable", False))
    l1_path = tcfg.get("l1_model_path")
    l2_path = tcfg.get("l2_model_path")
    top_k = int(tcfg.get("l2_top_k", 5) or 5)
    min_sim = tcfg.get("l2_min_similarity", None)
    min_sim = float(min_sim) if min_sim is not None else None
    return enabled, l1_path, l2_path, top_k, min_sim


# --- Taxonomy config helpers for backward/compat ---
def _taxonomy_overwrite_default(cfg: dict) -> bool:
    """Backward/compat: allow overwrite flag to live under `taxonomy:` too."""
    return bool((cfg.get("taxonomy", {}) or {}).get("overwrite_existing_taxonomy", True))


def _taxonomy_l1_min_confidence(cfg: dict) -> Optional[float]:
    """Optional confidence threshold for L1 predictions (from `taxonomy.l1_min_confidence`)."""
    v = (cfg.get("taxonomy", {}) or {}).get("l1_min_confidence", None)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _resolve_path(base_dir: Path, p: str) -> Path:
    """
    Generic resolver:
    - expanduser (~)
    - if relative, resolve against base_dir
    """
    pp = Path(p).expanduser()
    if not pp.is_absolute():
        pp = (base_dir / pp).resolve()
    return pp


@lru_cache(maxsize=16)
def _cached_joblib(abs_path: str, mtime: float, size: int):
    """
    Generic joblib cache (reload on overwrite due to mtime/size fingerprint).
    """
    import joblib  # local import to avoid unused import if taxonomy disabled
    return joblib.load(abs_path)


def _load_joblib_with_fingerprint(p: Path):
    st = p.stat()
    return _cached_joblib(str(p), float(st.st_mtime), int(st.st_size))


def _taxonomy_text_for_row(r: Dict[str, Any]) -> str:
    """
    Build the same kind of feature text used in training:
    name | desc | field | value (if present)
    We do NOT assume these columns always exist.
    """
    name = (
        clean(r.get("Proposed New Segment Name", ""))
        or clean(r.get("New Segment Name", ""))
        or clean(r.get("Segment Name", ""))
    )
    desc = clean(r.get("Segment Description", ""))
    field = clean(r.get("Field Name", "")) or clean(r.get("LiveRamp Field Name", ""))
    value = clean(r.get("Value Name", "")) or clean(r.get("LiveRamp Value Name", ""))
    parts = [x for x in [name, desc, field, value] if x]
    return " | ".join(parts)

def _apply_vertical_fallback_if_applicable(
    l1_model: Any,
    X: List[str],
    l1_pred: List[str],
    l1_conf: List[Optional[float]],
) -> Tuple[List[str], List[Optional[float]]]:
    """
    If l1_model is a TaxonomyL1Bundle-like object with:
      - vertical_model
      - b2b_label
      - vertical_min_confidence
    then override rows where main L1 predicted B2B and vertical is confident enough.
    """
    if not hasattr(l1_model, "vertical_model"):
        return l1_pred, l1_conf

    vmodel = getattr(l1_model, "vertical_model", None)
    if vmodel is None:
        return l1_pred, l1_conf

    b2b_label = str(getattr(l1_model, "b2b_label", "B2B Audience"))
    vmin = float(getattr(l1_model, "vertical_min_confidence", 0.55))

    # Only consider rows predicted as B2B
    idxs = [i for i, p in enumerate(l1_pred) if str(p) == b2b_label]
    if not idxs:
        return l1_pred, l1_conf

    Xb = [X[i] for i in idxs]

    # We need probabilities to compute confidence
    if not hasattr(vmodel, "predict_proba"):
        return l1_pred, l1_conf

    vproba = vmodel.predict_proba(Xb)
    vclasses = getattr(vmodel, "classes_", None)

    if vclasses is None:
        # Fallback: use predict() without confidence (won't meet threshold)
        return l1_pred, l1_conf

    # Choose best vertical label + confidence
    import numpy as np

    vproba = np.asarray(vproba)
    best_idx = vproba.argmax(axis=1)
    best_conf = vproba.max(axis=1)
    best_label = [str(vclasses[j]) for j in best_idx]

    # Override if confident enough
    for local_i, global_i in enumerate(idxs):
        if float(best_conf[local_i]) >= vmin:
            l1_pred[global_i] = best_label[local_i]
            l1_conf[global_i] = float(best_conf[local_i])

    return l1_pred, l1_conf

def _apply_taxonomy_if_enabled(
    base_dir: Path,
    cfg: dict,
    validated_scored: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Adds taxonomy predictions to validated_scored.

    Option A behavior (model can override generator taxonomy):
    1) Parse L1/L2 from "Proposed New Segment Name" FIRST (if present) to populate baseline.
    2) Run taxonomy models (L1 classifier + L2 retriever) to produce model predictions.
    3) If overwrite_existing_taxonomy=true (default), overwrite L1/L2 for ALL rows.
       If false, only fill rows where L1/L2 is missing.
       (Note: this flag can come from either taxonomy_model or taxonomy config section.)

    Never raises (won’t break streaming).
    """
    tax_log.info("[TAX] start")

    tcfg = cfg.get("taxonomy_model", {}) or {}
    enabled, l1_path, l2_path, top_k, min_sim = _taxonomy_config(cfg)
    if not enabled:
        tax_log.info("[TAX] disabled")
        return validated_scored, None

    if not l1_path or not l2_path:
        msg = "taxonomy_model.enable=true but l1_model_path/l2_model_path missing in config.yml"
        tax_log.error("[TAX] %s", msg)
        return validated_scored, msg

    if validated_scored is None or len(validated_scored) == 0:
        tax_log.info("[TAX] skip (no rows)")
        return validated_scored, None

    # output column names (configurable)
    out_l1 = str(tcfg.get("output_l1_col", "Predicted L1"))
    out_l2 = str(tcfg.get("output_l2_col", "Predicted L2"))
    out_l1_conf = str(tcfg.get("output_l1_conf_col", "Predicted L1 Confidence"))
    out_l2_sim = str(tcfg.get("output_l2_sim_col", "Predicted L2 Similarity"))

    # Option A switch: overwrite existing taxonomy or only fill blanks
    # Prefer taxonomy_model.overwrite_existing_taxonomy, but support older/newer config that stores it under `taxonomy:`
    overwrite_existing = bool(tcfg.get("overwrite_existing_taxonomy", _taxonomy_overwrite_default(cfg)))    # Optional L1 confidence threshold (stored under `taxonomy:`)
    l1_min_conf = _taxonomy_l1_min_confidence(cfg)

    try:
        out = validated_scored.copy()

        # Ensure output columns exist
        if out_l1 not in out.columns:
            out[out_l1] = ""
        if out_l2 not in out.columns:
            out[out_l2] = ""
        if out_l1_conf not in out.columns:
            out[out_l1_conf] = None
        if out_l2_sim not in out.columns:
            out[out_l2_sim] = None

        # ----------------------------------------
        # Parse-first: populate baseline from Proposed New Segment Name (debug/baseline)
        # ----------------------------------------
        proposed_col = "Proposed New Segment Name"
        if proposed_col in out.columns:

            def _parse_from_proposed_name(df: pd.DataFrame, col: str) -> pd.DataFrame:
                def parse_one(x):
                    if not isinstance(x, str):
                        return ("", "")
                    parts = [p.strip() for p in x.split(">")]
                    if len(parts) >= 4:
                        return (parts[1], parts[2])  # L1, L2
                    return ("", "")

                l1_l2 = df[col].apply(parse_one)
                parsed_l1 = l1_l2.apply(lambda t: t[0])
                parsed_l2 = l1_l2.apply(lambda t: t[1])

                # only fill blanks (do not overwrite existing values here)
                mask_l1_blank = df[out_l1].astype(str).str.strip().eq("")
                mask_l2_blank = df[out_l2].astype(str).str.strip().eq("")

                df.loc[mask_l1_blank, out_l1] = parsed_l1[mask_l1_blank].astype(str)
                df.loc[mask_l2_blank, out_l2] = parsed_l2[mask_l2_blank].astype(str)
                return df

            out = _parse_from_proposed_name(out, proposed_col)

        # Decide which rows to write back into (ALL vs only missing)
        if overwrite_existing:
            idx_target = out.index.tolist()
        else:
            mask_missing = (
                out[out_l1].astype(str).str.strip().eq("")
                | out[out_l2].astype(str).str.strip().eq("")
            )
            idx_target = out.index[mask_missing].tolist()

        if len(idx_target) == 0:
            tax_log.info("[TAX] skip (already filled) overwrite=%s", overwrite_existing)
            return out, None

        # Resolve model paths + load
        l1p = _resolve_path(base_dir, l1_path)
        l2p = _resolve_path(base_dir, l2_path)

        if not l1p.exists():
            msg = f"Taxonomy L1 model not found at: {l1p}"
            logger.error("[taxonomy] %s", msg)
            return out, msg
        if not l2p.exists():
            msg = f"Taxonomy L2 model not found at: {l2p}"
            logger.error("[taxonomy] %s", msg)
            return out, msg

        l1_model = _load_joblib_with_fingerprint(l1p)
        l2_model = _load_joblib_with_fingerprint(l2p)
        tax_log.info("[TAX] models loaded")
        # Build X ONLY for idx_target
        out_target = out.loc[idx_target].copy()
        if "text" in out_target.columns:
            X = out_target["text"].astype(str).tolist()
        else:
            X = [_taxonomy_text_for_row(r) for r in out_target.to_dict(orient="records")]

        # ----------------------------
        # L1 prediction (classifier)
        # ----------------------------
        try:
            l1_pred = l1_model.predict(X)

            # Default: overwrite target rows
            pred_series = pd.Series(l1_pred, index=idx_target).astype(str)

            l1_conf_vals: List[Optional[float]] = [None] * len(idx_target)
            conf_series = pd.Series(l1_conf_vals, index=idx_target)

            # If we can compute probabilities, enforce optional confidence threshold
            try:
                if hasattr(l1_model, "predict_proba"):
                    proba = l1_model.predict_proba(X)
                    l1_conf_vals = [float(p.max()) for p in proba]
                    conf_series = pd.Series(l1_conf_vals, index=idx_target)

                    # ✅ Apply learned vertical fallback (only affects rows predicted as B2B)
                    l1_pred_list = list(pred_series.astype(str).values)
                    l1_conf_list = list(conf_series.values)

                    l1_pred_list, l1_conf_list = _apply_vertical_fallback_if_applicable(
                        l1_model, X, l1_pred_list, l1_conf_list
                    )

                    pred_series = pd.Series(l1_pred_list, index=idx_target).astype(str)
                    conf_series = pd.Series(l1_conf_list, index=idx_target)

                    if l1_min_conf is not None:
                        mask_ok = conf_series >= float(l1_min_conf)
                    else:
                        mask_ok = pd.Series([True] * len(idx_target), index=idx_target)

                    if overwrite_existing:
                        out.loc[pred_series.index[mask_ok], out_l1] = pred_series[mask_ok]
                    else:
                        # only fill rows where L1 is blank
                        mask_blank = out.loc[idx_target, out_l1].astype(str).str.strip().eq("")
                        mask_final = mask_blank & mask_ok
                        out.loc[pred_series.index[mask_final], out_l1] = pred_series[mask_final]

                else:
                    if overwrite_existing:
                        out.loc[idx_target, out_l1] = pred_series
                    else:
                        mask_blank = out.loc[idx_target, out_l1].astype(str).str.strip().eq("")
                        out.loc[pred_series.index[mask_blank], out_l1] = pred_series[mask_blank]

            except Exception:
                # If proba fails, fall back to original behavior
                if overwrite_existing:
                    out.loc[idx_target, out_l1] = pred_series
                else:
                    mask_blank = out.loc[idx_target, out_l1].astype(str).str.strip().eq("")
                    out.loc[pred_series.index[mask_blank], out_l1] = pred_series[mask_blank]

            out.loc[idx_target, out_l1_conf] = conf_series
        except Exception as e:
            tax_log.warning("[TAX] L1 failed: %s", e)

        # ----------------------------
        # L2 retrieval (not a classifier)
        # ----------------------------
        try:
            hits = None

            if hasattr(l2_model, "query"):
                hits = l2_model.query(X, top_k=top_k)
            elif hasattr(l2_model, "search"):
                hits = l2_model.search(X, top_k=top_k)
            elif callable(l2_model):
                hits = l2_model(X, top_k=top_k)

            best_labels: List[str] = []
            best_scores: List[Optional[float]] = []

            if hits is None and hasattr(l2_model, "predict"):
                l2_pred = l2_model.predict(X)
                out.loc[idx_target, out_l2] = pd.Series(l2_pred, index=idx_target).astype(str)
                out.loc[idx_target, out_l2_sim] = pd.Series([None] * len(idx_target), index=idx_target)
                hits = "used_predict"

            if hits is None or hits == "used_predict":
                if hits is None:
                    logger.warning("[taxonomy] L2 retriever has no query/search interface; skipping L2 predictions")
                # if used_predict, already set; if none, leave existing values
            else:
                for row_hits in hits:
                    label = ""
                    score: Optional[float] = None

                    if isinstance(row_hits, dict):
                        label = str(row_hits.get("label") or row_hits.get("L2") or row_hits.get("name") or "")
                        sc = row_hits.get("score") or row_hits.get("similarity") or row_hits.get("sim")
                        score = float(sc) if sc is not None else None
                    elif isinstance(row_hits, list) and row_hits:
                        h0 = row_hits[0]
                        if isinstance(h0, dict):
                            label = str(h0.get("label") or h0.get("L2") or h0.get("name") or "")
                            sc = h0.get("score") or h0.get("similarity") or h0.get("sim")
                            score = float(sc) if sc is not None else None
                        elif isinstance(h0, (tuple, list)) and len(h0) >= 1:
                            label = str(h0[0])
                            if len(h0) >= 2 and h0[1] is not None:
                                try:
                                    score = float(h0[1])
                                except Exception:
                                    score = None

                    if min_sim is not None and score is not None and score < min_sim:
                        label = ""

                    best_labels.append(label)
                    best_scores.append(score)

                if len(best_labels) != len(idx_target):
                    best_labels = (best_labels + [""] * len(idx_target))[: len(idx_target)]
                if len(best_scores) != len(idx_target):
                    best_scores = (best_scores + [None] * len(idx_target))[: len(idx_target)]

                l2_series = pd.Series(best_labels, index=idx_target).astype(str)
                sim_series = pd.Series(best_scores, index=idx_target)

                if overwrite_existing:
                    out.loc[idx_target, out_l2] = l2_series
                else:
                    # only fill rows where L2 is blank
                    mask_blank_l2 = out.loc[idx_target, out_l2].astype(str).str.strip().eq("")
                    out.loc[l2_series.index[mask_blank_l2], out_l2] = l2_series[mask_blank_l2]

                out.loc[idx_target, out_l2_sim] = sim_series

        except Exception as e:
            tax_log.warning("[TAX] L2 failed: %s", e)

        tax_log.info("[TAX] done rows=%d overwrite=%s cols=(%s,%s)", len(idx_target), overwrite_existing, out_l1, out_l2)
        return out, None

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        tax_log.exception("[TAX] failed: %s", msg)
        return validated_scored, msg

# def _apply_taxonomy_if_enabled(
#     base_dir: Path,
#     cfg: dict,
#     validated_scored: pd.DataFrame,
# ) -> Tuple[pd.DataFrame, Optional[str]]:
#     """
#     Adds taxonomy predictions to validated_scored.

#     Behavior:
#     1) Parse L1/L2 from "Proposed New Segment Name" FIRST (if present).
#     2) Only run taxonomy models for rows where L1 and/or L2 is still missing.
#     3) Never raises (won’t break streaming).
#     """
#     logger.info("[taxonomy] ENTER _apply_taxonomy_if_enabled()")

#     tcfg = cfg.get("taxonomy_model", {}) or {}
#     enabled, l1_path, l2_path, top_k, min_sim = _taxonomy_config(cfg)
#     if not enabled:
#         logger.info("[taxonomy] disabled in config")
#         return validated_scored, None

#     if not l1_path or not l2_path:
#         msg = "taxonomy_model.enable=true but l1_model_path/l2_model_path missing in config.yml"
#         logger.error("[taxonomy] %s", msg)
#         return validated_scored, msg

#     if validated_scored is None or len(validated_scored) == 0:
#         logger.info("[taxonomy] no rows to label")
#         return validated_scored, None

    
#     out_l1 = str(tcfg.get("output_l1_col", "Predicted L1"))
#     out_l2 = str(tcfg.get("output_l2_col", "Predicted L2"))
#     out_l1_conf = str(tcfg.get("output_l1_conf_col", "Predicted L1 Confidence"))
#     out_l2_sim = str(tcfg.get("output_l2_sim_col", "Predicted L2 Similarity"))

#     try:
       
#         out = validated_scored.copy()

        
#         if out_l1 not in out.columns:
#             out[out_l1] = ""
#         if out_l2 not in out.columns:
#             out[out_l2] = ""
#         if out_l1_conf not in out.columns:
#             out[out_l1_conf] = None
#         if out_l2_sim not in out.columns:
#             out[out_l2_sim] = None

       
#         proposed_col = "Proposed New Segment Name"
#         if proposed_col in out.columns:

#             def _parse_from_proposed_name(df: pd.DataFrame, col: str) -> pd.DataFrame:
#                 def parse_one(x):
#                     if not isinstance(x, str):
#                         return ("", "")
#                     parts = [p.strip() for p in x.split(">")]
#                     if len(parts) >= 4:
#                         return (parts[1], parts[2])  
#                     return ("", "")

#                 l1_l2 = df[col].apply(parse_one)
                
#                 parsed_l1 = l1_l2.apply(lambda t: t[0])
#                 parsed_l2 = l1_l2.apply(lambda t: t[1])

#                 mask_l1_blank = df[out_l1].astype(str).str.strip().eq("")
#                 mask_l2_blank = df[out_l2].astype(str).str.strip().eq("")

#                 df.loc[mask_l1_blank, out_l1] = parsed_l1[mask_l1_blank].astype(str)
#                 df.loc[mask_l2_blank, out_l2] = parsed_l2[mask_l2_blank].astype(str)
#                 return df

#             out = _parse_from_proposed_name(out, proposed_col)

        
#         mask_missing = (
#             out[out_l1].astype(str).str.strip().eq("")
#             | out[out_l2].astype(str).str.strip().eq("")
#         )
#         idx_missing = out.index[mask_missing].tolist()

#         if len(idx_missing) == 0:
#             logger.info("[taxonomy] skipping model — all rows already have taxonomy from Proposed New Segment Name")
#             logger.info("[taxonomy] applied taxonomy cols: %s, %s (and optional conf/sim)", out_l1, out_l2)
#             return out, None

#         l1p = _resolve_path(base_dir, l1_path)
#         l2p = _resolve_path(base_dir, l2_path)

#         if not l1p.exists():
#             msg = f"Taxonomy L1 model not found at: {l1p}"
#             logger.error("[taxonomy] %s", msg)
#             return out, msg
#         if not l2p.exists():
#             msg = f"Taxonomy L2 model not found at: {l2p}"
#             logger.error("[taxonomy] %s", msg)
#             return out, msg

#         l1_model = _load_joblib_with_fingerprint(l1p)
#         l2_model = _load_joblib_with_fingerprint(l2p)

#         out_missing = out.loc[idx_missing].copy()
#         if "text" in out_missing.columns:
#             X = out_missing["text"].astype(str).tolist()
#         else:
#             X = [_taxonomy_text_for_row(r) for r in out_missing.to_dict(orient="records")]

#         try:
#             l1_pred = l1_model.predict(X)
#             out.loc[idx_missing, out_l1] = pd.Series(l1_pred, index=idx_missing).astype(str)

#             l1_conf_vals: List[Optional[float]] = [None] * len(idx_missing)
#             try:
#                 if hasattr(l1_model, "predict_proba"):
#                     proba = l1_model.predict_proba(X)
#                     l1_conf_vals = [float(p.max()) for p in proba]
#             except Exception:
#                 pass

#             out.loc[idx_missing, out_l1_conf] = pd.Series(l1_conf_vals, index=idx_missing)
#         except Exception as e:
#             logger.warning("[taxonomy] L1 prediction failed: %s", e)

#         try:
#             hits = None

#             if hasattr(l2_model, "query"):
#                 hits = l2_model.query(X, top_k=top_k)
#             elif hasattr(l2_model, "search"):
#                 hits = l2_model.search(X, top_k=top_k)
#             elif callable(l2_model):
#                 hits = l2_model(X, top_k=top_k)

#             best_labels: List[str] = []
#             best_scores: List[Optional[float]] = []

#             if hits is None and hasattr(l2_model, "predict"):
#                 l2_pred = l2_model.predict(X)
#                 out.loc[idx_missing, out_l2] = pd.Series(l2_pred, index=idx_missing).astype(str)
#                 out.loc[idx_missing, out_l2_sim] = pd.Series([None] * len(idx_missing), index=idx_missing)
#                 hits = "used_predict"

#             if hits is None or hits == "used_predict":
#                 if hits is None:
#                     logger.warning("[taxonomy] L2 retriever has no query/search interface; skipping L2 predictions")
#                 if hits is None:
#                     out.loc[idx_missing, out_l2] = out.loc[idx_missing, out_l2].astype(str)
#             else:
#                 for row_hits in hits:
#                     label = ""
#                     score: Optional[float] = None

#                     if isinstance(row_hits, dict):
#                         label = str(row_hits.get("label") or row_hits.get("L2") or row_hits.get("name") or "")
#                         sc = row_hits.get("score") or row_hits.get("similarity") or row_hits.get("sim")
#                         score = float(sc) if sc is not None else None
#                     elif isinstance(row_hits, list) and row_hits:
#                         h0 = row_hits[0]
#                         if isinstance(h0, dict):
#                             label = str(h0.get("label") or h0.get("L2") or h0.get("name") or "")
#                             sc = h0.get("score") or h0.get("similarity") or h0.get("sim")
#                             score = float(sc) if sc is not None else None
#                         elif isinstance(h0, (tuple, list)) and len(h0) >= 1:
#                             label = str(h0[0])
#                             if len(h0) >= 2 and h0[1] is not None:
#                                 try:
#                                     score = float(h0[1])
#                                 except Exception:
#                                     score = None

#                     if min_sim is not None and score is not None and score < min_sim:
#                         label = ""

#                     best_labels.append(label)
#                     best_scores.append(score)

#                 if len(best_labels) != len(idx_missing):
#                     best_labels = (best_labels + [""] * len(idx_missing))[: len(idx_missing)]
#                 if len(best_scores) != len(idx_missing):
#                     best_scores = (best_scores + [None] * len(idx_missing))[: len(idx_missing)]

#                 out.loc[idx_missing, out_l2] = pd.Series(best_labels, index=idx_missing).astype(str)
#                 out.loc[idx_missing, out_l2_sim] = pd.Series(best_scores, index=idx_missing)

#         except Exception as e:
#             logger.warning("[taxonomy] L2 retrieval failed: %s", e)

#         logger.info("[taxonomy] applied taxonomy cols: %s, %s (and optional conf/sim)", out_l1, out_l2)
#         return out, None

#     except Exception as e:
#         msg = f"{type(e).__name__}: {e}"
#         logger.exception("[taxonomy] FAILED: %s", msg)
#         return validated_scored, msg

def _resolve_allowed_categories(cfg: dict, allowed_categories: Optional[List[str]]) -> List[str]:
    gen = cfg.get("generation", {}) or {}
    default = gen.get("allowed_categories_default", []) or []
    # UI override wins if provided and non-empty
    if allowed_categories:
        return [str(x).strip() for x in allowed_categories if str(x).strip()]
    return [str(x).strip() for x in default if str(x).strip()]


# ----------------------------
# Pipeline runner (non-streaming)
# ----------------------------
def run_pipeline(
    base_dir: Path,
    max_rows: Optional[int] = None,
    allowed_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    cfg = load_config(base_dir)

    _log_pipeline_start(cfg, max_rows=max_rows, allowed_categories=allowed_categories)

    use_gate = bool((cfg.get("generation", {}) or {}).get("use_allowed_categories_gating", False))
    effective_allowed = _resolve_allowed_categories(cfg, allowed_categories) if use_gate else None
    proposals_df, coverage_df = generate_proposals(cfg, allowed_categories=effective_allowed)

    # Web assistance (non-streaming path)
    _web_cfg_ns = cfg.get("llm_web_assistance", {}) or {}
    _web_names_ns: set = set()
    if bool(_web_cfg_ns.get("enable", False)) and proposals_df is not None and not proposals_df.empty:
        try:
            from src.web_assistance import generate_web_assisted_segments  # noqa: PLC0415
            _input_dir_ns = Path(cfg["paths"]["input_dir"])
            _A_ns = pd.read_csv(_input_dir_ns / cfg["files"]["input_a"], dtype=str, keep_default_na=False)
            _existing_ns = _A_ns["Segment Name"].tolist() if "Segment Name" in _A_ns.columns else []
            web_df_ns = generate_web_assisted_segments(
                cfg,
                existing_catalog_names=_existing_ns,
                ollama_url=str(_web_cfg_ns.get("ollama_url", "http://host.docker.internal:11434/api/generate")),
                model=str(_web_cfg_ns.get("model", "llama3.1")),
                max_segments=int(_web_cfg_ns.get("max_segments", 20)),
                max_search_queries=int(_web_cfg_ns.get("max_search_queries", 5)),
                timeout=int(_web_cfg_ns.get("timeout_seconds", 60)),
            )
            if web_df_ns is not None and not web_df_ns.empty:
                _web_names_ns = set(web_df_ns["Proposed New Segment Name"].tolist())
                proposals_df = pd.concat([proposals_df, web_df_ns], ignore_index=True)
        except Exception as _e_ns:
            logger.error("Web assistance (non-streaming) failed: %s", _e_ns)

    # 🚨 HARD STOP: no proposals → skip validators entirely (non-streaming)
    if proposals_df is None or proposals_df.empty:
        empty_validated = pd.DataFrame(columns=[
            "Competitor Provider",
            "Competitor Segment Name",
            "Competitor Segment ID",
            "Proposed New Segment Name",
            "Non Derived Segments utilized",
            "Composition Similarity",
            "Closest Cybba Segment",
            "Closest Cybba Similarity",
            "Taxonomy",
            "Segment Description",
            "uniqueness_score",
            "rank_score",
        ])

        template_df = load_output_template(cfg)
        final_df = build_final_from_template(empty_validated, template_df)

        summary = {
            "total_proposals": 0,
            "validated_total": 0,
            "validated_generated": 0,
            "cap_applied": int(max_rows) if isinstance(max_rows, int) else None,
            "covered": int(len(coverage_df)) if coverage_df is not None else 0,
            "duplicates_within_output_normalized": 0,
            "dropped_naming": 0,
            "dropped_underived_only": 0,
            "dropped_net_new": 0,
            "collisions_resolved": 0,
            "uniqueness_stats": safe_quantiles(pd.Series(dtype=float)),
            "rank_stats": safe_quantiles(pd.Series(dtype=float)),
            "pricing_enabled": bool((cfg.get("pricing_model", {}) or {}).get("enable", False)),
            "pricing_error": None,
            "taxonomy_enabled": bool((cfg.get("taxonomy_model", {}) or {}).get("enable", False)),
            "taxonomy_error": None,
            "reason": "No proposals generated (category gating / filtering)",
        }
        _log_pipeline_summary(summary)
        return {
            "cfg": cfg,
            "proposals_df": pd.DataFrame(columns=[]),
            "coverage_df": coverage_df if coverage_df is not None else pd.DataFrame(columns=[]),
            "validated_df": empty_validated,
            "final_df": final_df,
            "report": None,
            "summary": summary,
            }

    input_dir = Path(cfg["paths"]["input_dir"])
    A = pd.read_csv(input_dir / cfg["files"]["input_a"], dtype=str, keep_default_na=False)
    B = pd.read_csv(input_dir / cfg["files"]["input_b"], dtype=str, keep_default_na=False)

    validated_df, report = validate_and_prepare(
        proposals_df,
        underived_df=B,
        distributed_df=A,
        cfg=cfg,
    )

    # Re-attach web_assisted flag (non-streaming path)
    if _web_names_ns:
        validated_df["web_assisted"] = validated_df["Proposed New Segment Name"].isin(_web_names_ns)
    else:
        validated_df["web_assisted"] = False

    val_log.info(
    "[VAL] done kept=%d dropped(naming=%d underived=%d net_new=%d) collisions=%d",
    len(validated_df),
    int(getattr(report, "dropped_naming", 0)),
    int(getattr(report, "dropped_underived_only", 0)),
    int(getattr(report, "dropped_net_new", 0)),
    int(getattr(report, "collisions_resolved", 0)),
    )

    validated_scored_full = add_uniqueness_and_rank(validated_df).copy()
    validated_scored_full = _sort_for_generation(validated_scored_full)

    validated_total = int(len(validated_scored_full))
    validated_scored = _apply_cap(validated_scored_full, max_rows)

    enable_desc = bool(cfg.get("descriptions", {}).get("enable", False))
    if enable_desc and len(validated_scored) > 0:
        desc_cfg = cfg.get("descriptions", {}) or {}
        descs = generate_descriptions(
            validated_scored,
            model=desc_cfg.get("model", "llama3.1"),
            batch_size=int(desc_cfg.get("batch_size", 20) or 20),
            cache=DESCRIPTION_CACHE,
            ollama_url=str(desc_cfg.get("ollama_url", "http://localhost:11434/api/generate")),
            timeout=int(desc_cfg.get("timeout_seconds", 60) or 60),
        )
        validated_scored["Segment Description"] = pd.Series(descs, index=validated_scored.index)
    else:
        validated_scored["Segment Description"] = ""

    # ✅ Apply taxonomy (adds columns on validated_scored; never crashes)
    validated_scored, taxonomy_error = _apply_taxonomy_if_enabled(base_dir, cfg, validated_scored)

    template_df = load_output_template(cfg)
    final_df = build_final_from_template(validated_scored, template_df)

    final_df, pricing_error = _apply_pricing_if_enabled(base_dir, cfg, validated_scored, final_df)

    if "Proposed New Segment Name" in validated_scored.columns:
        norm = validated_scored["Proposed New Segment Name"].map(normalize_name)
        dup_within_norm = int(norm.duplicated().sum())
    else:
        dup_within_norm = 0

    uniq_stats = safe_quantiles(validated_scored.get("uniqueness_score", pd.Series(dtype=float)))
    rank_stats = safe_quantiles(validated_scored.get("rank_score", pd.Series(dtype=float)))

    summary = {
        "total_proposals": int(len(proposals_df)),
        "validated_total": validated_total,
        "validated_generated": int(len(validated_scored)),
        "cap_applied": int(max_rows) if isinstance(max_rows, int) else None,
        "covered": int(len(coverage_df)),
        "duplicates_within_output_normalized": int(dup_within_norm),
        "dropped_naming": int(getattr(report, "dropped_naming", 0)),
        "dropped_underived_only": int(getattr(report, "dropped_underived_only", 0)),
        "dropped_net_new": int(getattr(report, "dropped_net_new", 0)),
        "collisions_resolved": int(getattr(report, "collisions_resolved", 0)),
        "uniqueness_stats": uniq_stats,
        "rank_stats": rank_stats,
        "pricing_enabled": bool((cfg.get("pricing_model", {}) or {}).get("enable", False)),
        "pricing_error": pricing_error,
        "taxonomy_enabled": bool((cfg.get("taxonomy_model", {}) or {}).get("enable", False)),
        "taxonomy_error": taxonomy_error,
    }

    return {
        "cfg": cfg,
        "proposals_df": proposals_df,
        "coverage_df": coverage_df,
        "validated_df": validated_scored,
        "final_df": final_df,
        "report": report,
        "summary": summary,
    }


# ----------------------------
# Pipeline runner (streaming)
# ----------------------------
def run_pipeline_stream(
    base_dir: Path,
    max_rows: Optional[int] = None,
    allowed_categories: Optional[List[str]] = None,
    overrides: Optional[dict] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Generator that yields:
      {"event": "summary", "data": {...}}
      {"event": "row",     "data": {...}}
      {"event": "done",    "data": {...}}
    """
    cfg = load_config(base_dir)

    overrides = overrides or {}

    if overrides.get("enable_descriptions") is not None:
        cfg.setdefault("descriptions", {})["enable"] = bool(overrides["enable_descriptions"])

    if overrides.get("enable_pricing") is not None:
        cfg.setdefault("pricing_model", {})["enable"] = bool(overrides["enable_pricing"])

    if overrides.get("enable_taxonomy") is not None:
        cfg.setdefault("taxonomy_model", {})["enable"] = bool(overrides["enable_taxonomy"])

    if overrides.get("enable_coverage") is not None:
        cfg.setdefault("coverage", {})["enable"] = bool(overrides["enable_coverage"])
    if overrides.get("enable_llm_generation") is not None:
        cfg.setdefault("llm_generation", {})["enable"] = bool(overrides["enable_llm_generation"])

    if overrides.get("enable_llm_web_assistance") is not None:
        cfg.setdefault("llm_web_assistance", {})["enable"] = bool(overrides["enable_llm_web_assistance"])

    _log_pipeline_start(cfg, max_rows=max_rows, allowed_categories=allowed_categories)

    use_gate = bool((cfg.get("generation", {}) or {}).get("use_allowed_categories_gating", False))
    effective_allowed = _resolve_allowed_categories(cfg, allowed_categories) if use_gate else None

    # Tell UI generation started immediately
    yield {
        "event": "summary",
        "data": {"phase": "generating_proposals"}
    }

    # Time the expensive step
# Time the expensive step
    with _Timer(gen_log, "generate_proposals"):

        # Thread-safe progress pipe from generator -> SSE
        q: "queue.Queue[dict]" = queue.Queue(maxsize=1000)
        done_evt = threading.Event()

        result_holder: Dict[str, Any] = {"proposals_df": None, "coverage_df": None, "error": None}

        def progress_cb(payload: dict) -> None:
            # payload is from generate_proposals(); do not block worker thread
            try:
                q.put_nowait(payload)
            except Exception:
                # If queue is full, drop progress updates (never break generation)
                pass

        def worker() -> None:
            try:
                # Preferred call: new signature supports progress_cb/progress_every
                proposals_df, coverage_df = generate_proposals(
                    cfg,
                    allowed_categories=effective_allowed,
                    progress_cb=progress_cb,
                    progress_every=50,  # tune: lower = more updates
                )
                result_holder["proposals_df"] = proposals_df
                result_holder["coverage_df"] = coverage_df
            except TypeError:
                # Backward-compatible fallback if generate_proposals() doesn't accept these args
                proposals_df, coverage_df = generate_proposals(
                    cfg,
                    allowed_categories=effective_allowed,
                )
                result_holder["proposals_df"] = proposals_df
                result_holder["coverage_df"] = coverage_df
            except Exception as e:
                result_holder["error"] = e
            finally:
                done_evt.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # Drain progress while worker runs
        last_sent_pct: Optional[int] = None

        while not done_evt.is_set():
            try:
                msg = q.get(timeout=0.25)
            except queue.Empty:
                continue

            # Convert generate_proposals progress to 0-100
            cur = int(msg.get("current", 0) or 0)
            total = int(msg.get("total", 0) or 0)
            pct = int((cur / total) * 100) if total > 0 else 0

            # Avoid spamming identical % values
            if last_sent_pct == pct:
                continue
            last_sent_pct = pct

            yield {
                "event": "summary",
                "data": {
                    "phase": "generating_proposals",
                    "progress": pct,            # ✅ 0-100
                    "current": cur,
                    "total": total,
                    "kept": int(msg.get("kept", 0) or 0),
                    "covered": int(msg.get("covered", 0) or 0),
                    "blocked_abm": int(msg.get("blocked_abm", 0) or 0),
                    "blocked_entity": int(msg.get("blocked_entity", 0) or 0),
                    "blocked_category": int(msg.get("blocked_category", 0) or 0),
                    "allowed_categories": effective_allowed or [],
                },
            }

        # Worker finished: raise if it failed
        if result_holder["error"] is not None:
            raise result_holder["error"]

        proposals_df = result_holder["proposals_df"]
        coverage_df = result_holder["coverage_df"]

        # Ensure we end phase at 100 for this step
        yield {
            "event": "summary",
            "data": {
                "phase": "generating_proposals",
                "progress": 100,
                "allowed_categories": effective_allowed or [],
            },
        }

    # ── Web Assistance (optional) ──────────────────────────────────────────
    _web_cfg = cfg.get("llm_web_assistance", {}) or {}
    if bool(_web_cfg.get("enable", False)):
        try:
            from src.web_assistance import generate_web_assisted_segments  # noqa: PLC0415

            yield {"event": "summary", "data": {"phase": "web_assistance"}}
            logger.info("Web assistance enabled — fetching novel segments from web sources")

            # Load existing catalog names to check novelty
            try:
                _input_dir = Path(cfg["paths"]["input_dir"])
                _A = pd.read_csv(_input_dir / cfg["files"]["input_a"], dtype=str, keep_default_na=False)
                _existing_names = _A["Segment Name"].tolist() if "Segment Name" in _A.columns else []
            except Exception:
                _existing_names = []

            web_df = generate_web_assisted_segments(
                cfg,
                existing_catalog_names=_existing_names,
                ollama_url=str(_web_cfg.get("ollama_url", "http://host.docker.internal:11434/api/generate")),
                model=str(_web_cfg.get("model", "llama3.1")),
                max_segments=int(_web_cfg.get("max_segments", 20)),
                max_search_queries=int(_web_cfg.get("max_search_queries", 5)),
                timeout=int(_web_cfg.get("timeout_seconds", 60)),
            )

            if web_df is not None and not web_df.empty:
                # Save web_assisted flag before concat (preserve through validation)
                web_names = set(web_df["Proposed New Segment Name"].tolist())
                proposals_df = pd.concat([proposals_df, web_df], ignore_index=True)
                logger.info("Web assistance: merged %d web-assisted proposals (total now %d)", len(web_df), len(proposals_df))
                yield {"event": "summary", "data": {"phase": "web_assistance", "web_segments_added": len(web_df)}}
            else:
                web_names = set()
                logger.info("Web assistance: no web-assisted proposals were added")
        except Exception as _e:
            web_names = set()
            logger.error("Web assistance failed (skipping): %s", _e, exc_info=True)
    else:
        web_names = set()

    # 🚨 HARD STOP: no proposals → skip validators entirely (streaming)
    if proposals_df is None or proposals_df.empty:
        summary_early = {
            "total_proposals": 0,
            "validated_total": 0,
            "validated_generated": 0,
            "cap_applied": int(max_rows) if isinstance(max_rows, int) else None,
            "covered": int(len(coverage_df)) if coverage_df is not None else 0,
            "dropped_naming": 0,
            "dropped_underived_only": 0,
            "dropped_net_new": 0,
            "collisions_resolved": 0,
            "phase": "done",
            "pricing_enabled": bool((cfg.get("pricing_model", {}) or {}).get("enable", False)),
            "taxonomy_enabled": bool((cfg.get("taxonomy_model", {}) or {}).get("enable", False)),
            "allowed_categories": effective_allowed or [],
            "reason": "No proposals generated (category gating / filtering)",
        }

        _log_pipeline_summary(summary_early)
        yield {"event": "summary", "data": summary_early}
        yield {"event": "done", "data": {"summary": summary_early, "rows": [], "final_rows": []}}
        return

    input_dir = Path(cfg["paths"]["input_dir"])
    A = pd.read_csv(input_dir / cfg["files"]["input_a"], dtype=str, keep_default_na=False)
    B = pd.read_csv(input_dir / cfg["files"]["input_b"], dtype=str, keep_default_na=False)

    validated_df, report = validate_and_prepare(
        proposals_df,
        underived_df=B,
        distributed_df=A,
        cfg=cfg,
    )

    # Re-attach web_assisted flag (validators may drop unknown columns)
    if web_names:
        validated_df["web_assisted"] = validated_df["Proposed New Segment Name"].isin(web_names)
    else:
        validated_df["web_assisted"] = False

    validated_scored_full = add_uniqueness_and_rank(validated_df).copy()
    validated_scored_full = _sort_for_generation(validated_scored_full)

    validated_total = int(len(validated_scored_full))
    validated_scored = _apply_cap(validated_scored_full, max_rows)

    # ✅ apply taxonomy before per-row streaming so each row includes taxonomy cols
    #validated_scored, taxonomy_error = _apply_taxonomy_if_enabled(base_dir, cfg, validated_scored)
    taxonomy_error = None  # keep streaming behavior unchanged

    enable_desc = bool(cfg.get("descriptions", {}).get("enable", False))
    desc_model = cfg.get("descriptions", {}).get("model", "llama3.1")
    desc_cfg = cfg.get("descriptions", {}) or {}
    desc_ollama_url = str(desc_cfg.get("ollama_url", "http://localhost:11434/api/generate"))
    desc_timeout_s = int(desc_cfg.get("timeout_seconds", 60) or 60)

    summary_early = {
        "total_proposals": int(len(proposals_df)),
        "validated_total": int(validated_total),
        "validated_generated": int(len(validated_scored)),
        "cap_applied": int(max_rows) if isinstance(max_rows, int) else None,
        "covered": int(len(coverage_df)),
        "dropped_naming": int(getattr(report, "dropped_naming", 0)),
        "dropped_underived_only": int(getattr(report, "dropped_underived_only", 0)),
        "dropped_net_new": int(getattr(report, "dropped_net_new", 0)),
        "collisions_resolved": int(getattr(report, "collisions_resolved", 0)),
        "phase": "streaming_rows",
        "pricing_enabled": bool((cfg.get("pricing_model", {}) or {}).get("enable", False)),
        "taxonomy_enabled": bool((cfg.get("taxonomy_model", {}) or {}).get("enable", False)),
        "allowed_categories": effective_allowed or [],
    }
    yield {"event": "summary", "data": summary_early}

    streamed_rows: List[dict] = []

    for _, row in validated_scored.iterrows():
        r = row.to_dict()

        seg_name = clean(r.get("Proposed New Segment Name", ""))
        comps = clean(r.get("Non Derived Segments utilized", ""))

        cache_key = f"{seg_name}|{comps}"
        if cache_key in DESCRIPTION_CACHE:
            desc = DESCRIPTION_CACHE[cache_key]
        else:
            if enable_desc:
                desc = generate_segment_description(
                segment_name=seg_name,
                components=comps,
                model=desc_model,
                ollama_url=desc_ollama_url,
                timeout=desc_timeout_s,
            )
            else:
                desc = ""
            DESCRIPTION_CACHE[cache_key] = desc

        r["Segment Description"] = desc
        streamed_rows.append(r)

        yield {"event": "row", "data": r}

    if not streamed_rows:
        validated_scored2 = validated_scored.copy()
    else:
        validated_scored2 = pd.DataFrame(streamed_rows)

    if "Proposed New Segment Name" in validated_scored2.columns:
        norm = validated_scored2["Proposed New Segment Name"].map(normalize_name)
        dup_within_norm = int(norm.duplicated().sum())
    else:
        dup_within_norm = 0

    uniq_stats = safe_quantiles(validated_scored2.get("uniqueness_score", pd.Series(dtype=float)))
    rank_stats = safe_quantiles(validated_scored2.get("rank_score", pd.Series(dtype=float)))

    # ✅ Apply taxonomy at the end so it sees final descriptions too
    validated_scored2, taxonomy_error = _apply_taxonomy_if_enabled(base_dir, cfg, validated_scored2)

    template_df = load_output_template(cfg)
    final_df = build_final_from_template(validated_scored2, template_df)

    final_df, pricing_error = _apply_pricing_if_enabled(base_dir, cfg, validated_scored2, final_df)

    summary_final = {
        **summary_early,
        "duplicates_within_output_normalized": int(dup_within_norm),
        "uniqueness_stats": uniq_stats,
        "rank_stats": rank_stats,
        "phase": "done",
        "pricing_error": pricing_error,
        "taxonomy_error": taxonomy_error,
    }

    yield {
        "event": "done",
        "data": {
            "summary": summary_final,
            "rows": _df_to_rows(validated_scored2),
            "final_rows": _df_to_rows(final_df),
        },
    }


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")