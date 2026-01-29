# backend/src/pipeline.py
from __future__ import annotations

import io
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import yaml

from llm_descriptions import generate_descriptions, generate_segment_description
from segment_expansion_model import generate_proposals
from validators import validate_and_prepare

# Pricing model (trained .joblib inference)
from pricing_model import PRICE_COLS, PricingDefaults, load_pricing_model, predict_prices

# ----------------------------
# Logging (prints may be buffered; logging is safer)
# ----------------------------
logger = logging.getLogger("pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

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


def _df_to_rows(df: pd.DataFrame) -> List[dict]:
    """
    Convert DataFrame to JSON-safe rows (avoid NaN/NA).
    """
    return df.replace({pd.NA: None}).where(pd.notnull(df), None).to_dict(orient="records")


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
    logger.info("[pricing] ENTER _apply_pricing_if_enabled()")

    enabled, model_path, defaults = _pricing_config(cfg)
    if not enabled:
        logger.info("[pricing] disabled in config")
        return final_df, None

    if not model_path:
        msg = "pricing_model.enable=true but pricing_model.model_path is missing in config.yml"
        logger.error("[pricing] %s", msg)
        return final_df, msg

    if validated_scored is None or len(validated_scored) == 0:
        logger.info("[pricing] no rows to price")
        return final_df, None

    try:
        resolved = _resolve_model_path(base_dir, model_path)
        if not resolved.exists():
            msg = f"Pricing model not found at: {resolved}"
            logger.error("[pricing] %s", msg)
            return final_df, msg

        st = resolved.stat()
        logger.info(
            "[pricing] model_path resolved to: %s (exists=True, mtime=%s, size=%s)",
            resolved,
            float(st.st_mtime),
            int(st.st_size),
        )

        # ✅ always reloads after retrain overwrite (mtime/size changes)
        model = _load_pricing_model_with_fingerprint(resolved)

        logger.info("[pricing] calling predict_prices() on %d rows", len(validated_scored))
        prices_df = predict_prices(model, validated_scored, defaults=defaults)

        # --- diagnostics that will actually show up ---
        try:
            desc = prices_df.describe(include="all")
            logger.info("[pricing] describe:\n%s", desc.to_string())
        except Exception as e:
            logger.warning("[pricing] describe() failed: %s", e)

        try:
            nonzero_any = (prices_df != 0).any()
            logger.info("[pricing] any nonzero?:\n%s", nonzero_any.to_string())
            logger.info("[pricing] head:\n%s", prices_df.head(5).to_string(index=False))
            try:
                identical = bool((prices_df.nunique(dropna=False) <= 1).all())
                logger.info("[pricing] all rows identical?: %s", identical)
            except Exception:
                pass
        except Exception as e:
            logger.warning("[pricing] nonzero/head diagnostics failed: %s", e)

        out = final_df.copy()
        for c in PRICE_COLS:
            if c in out.columns and c in prices_df.columns:
                out[c] = prices_df[c]

        return out, None

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("[pricing] FAILED: %s", msg)
        # Do NOT crash streaming; surface error in summary
        return final_df, msg


# ----------------------------
# Pipeline runner (non-streaming)
# ----------------------------
def run_pipeline(base_dir: Path, max_rows: Optional[int] = None) -> Dict[str, Any]:
    cfg = load_config(base_dir)

    proposals_df, coverage_df = generate_proposals(cfg)

    input_dir = Path(cfg["paths"]["input_dir"])
    A = pd.read_csv(input_dir / cfg["files"]["input_a"], dtype=str, keep_default_na=False)
    B = pd.read_csv(input_dir / cfg["files"]["input_b"], dtype=str, keep_default_na=False)

    validated_df, report = validate_and_prepare(
        proposals_df,
        underived_df=B,
        distributed_df=A,
        cfg=cfg,
    )

    validated_scored_full = add_uniqueness_and_rank(validated_df).copy()
    validated_scored_full = _sort_for_generation(validated_scored_full)

    validated_total = int(len(validated_scored_full))
    validated_scored = _apply_cap(validated_scored_full, max_rows)

    enable_desc = bool(cfg.get("descriptions", {}).get("enable", False))
    if enable_desc and len(validated_scored) > 0:
        descs = generate_descriptions(
            validated_scored,
            model=cfg["descriptions"].get("model", "llama3.1"),
            batch_size=int(cfg["descriptions"].get("batch_size", 20)),
            cache=DESCRIPTION_CACHE,
        )
        validated_scored["Segment Description"] = pd.Series(descs, index=validated_scored.index)
    else:
        validated_scored["Segment Description"] = ""

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
def run_pipeline_stream(base_dir: Path, max_rows: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    Generator that yields:
      {"event": "summary", "data": {...}}
      {"event": "row",     "data": {...}}
      {"event": "done",    "data": {...}}
    """
    cfg = load_config(base_dir)

    proposals_df, coverage_df = generate_proposals(cfg)

    input_dir = Path(cfg["paths"]["input_dir"])
    A = pd.read_csv(input_dir / cfg["files"]["input_a"], dtype=str, keep_default_na=False)
    B = pd.read_csv(input_dir / cfg["files"]["input_b"], dtype=str, keep_default_na=False)

    validated_df, report = validate_and_prepare(
        proposals_df,
        underived_df=B,
        distributed_df=A,
        cfg=cfg,
    )

    validated_scored_full = add_uniqueness_and_rank(validated_df).copy()
    validated_scored_full = _sort_for_generation(validated_scored_full)

    validated_total = int(len(validated_scored_full))
    validated_scored = _apply_cap(validated_scored_full, max_rows)

    enable_desc = bool(cfg.get("descriptions", {}).get("enable", False))
    desc_model = cfg.get("descriptions", {}).get("model", "llama3.1")

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
