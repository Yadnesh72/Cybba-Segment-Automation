# backend/src/pipeline.py
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pandas as pd
import yaml

from segment_expansion_model import generate_proposals
from validators import validate_and_prepare
from llm_descriptions import generate_descriptions, generate_segment_description


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

    # Fill remaining columns
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


# ----------------------------
# Pipeline runner (non-streaming)
# ----------------------------
def run_pipeline(base_dir: Path) -> Dict[str, Any]:
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

    validated_scored = add_uniqueness_and_rank(validated_df).copy()

    # LLM descriptions (optional toggle via config)
    if cfg.get("descriptions", {}).get("enable", False):
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

    # summary
    if "Proposed New Segment Name" in validated_scored.columns:
        norm = validated_scored["Proposed New Segment Name"].map(normalize_name)
        dup_within_norm = int(norm.duplicated().sum())
    else:
        dup_within_norm = 0

    uniq_stats = safe_quantiles(validated_scored.get("uniqueness_score", pd.Series(dtype=float)))
    rank_stats = safe_quantiles(validated_scored.get("rank_score", pd.Series(dtype=float)))

    summary = {
        "total_proposals": int(len(proposals_df)),
        "validated": int(len(validated_scored)),
        "covered": int(len(coverage_df)),
        "duplicates_within_output_normalized": int(dup_within_norm),
        "dropped_naming": int(getattr(report, "dropped_naming", 0)),
        "dropped_underived_only": int(getattr(report, "dropped_underived_only", 0)),
        "dropped_net_new": int(getattr(report, "dropped_net_new", 0)),
        "collisions_resolved": int(getattr(report, "collisions_resolved", 0)),
        "uniqueness_stats": uniq_stats,
        "rank_stats": rank_stats,
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
def run_pipeline_stream(base_dir: Path) -> Iterator[Dict[str, Any]]:
    """
    Generator that yields:
      {"event": "summary", "data": {...}}
      {"event": "row",     "data": {...}}
      {"event": "done",    "data": {...}}

    Use this with SSE / StreamingResponse so the frontend can append rows while LLM runs.
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

    validated_scored = add_uniqueness_and_rank(validated_df).copy()

    enable_desc = bool(cfg.get("descriptions", {}).get("enable", False))
    model = cfg.get("descriptions", {}).get("model", "llama3.1")

    # Emit an early summary so UI can render immediately
    summary_early = {
        "total_proposals": int(len(proposals_df)),
        "validated": int(len(validated_scored)),
        "covered": int(len(coverage_df)),
        "dropped_naming": int(getattr(report, "dropped_naming", 0)),
        "dropped_underived_only": int(getattr(report, "dropped_underived_only", 0)),
        "dropped_net_new": int(getattr(report, "dropped_net_new", 0)),
        "collisions_resolved": int(getattr(report, "collisions_resolved", 0)),
        "phase": "streaming_rows",
    }
    yield {"event": "summary", "data": summary_early}

    streamed_rows: List[dict] = []

    # Stream each row as we generate (or skip) descriptions
    for _, row in validated_scored.iterrows():
        r = row.to_dict()

        seg_name = clean(r.get("Proposed New Segment Name", ""))
        comps = clean(r.get("Non Derived Segments utilized", ""))

        # cache hit?
        cache_key = f"{seg_name}|{comps}"
        if cache_key in DESCRIPTION_CACHE:
            desc = DESCRIPTION_CACHE[cache_key]
        else:
            if enable_desc:
                desc = generate_segment_description(
                    segment_name=seg_name,
                    components=comps,
                    model=model,
                )
            else:
                desc = ""
            DESCRIPTION_CACHE[cache_key] = desc

        r["Segment Description"] = desc
        streamed_rows.append(r)

        # emit row immediately
        yield {"event": "row", "data": r}

    # Final dataframes after streaming
    validated_scored2 = pd.DataFrame(streamed_rows)

    # Final summary stats once complete
    if "Proposed New Segment Name" in validated_scored2.columns:
        norm = validated_scored2["Proposed New Segment Name"].map(normalize_name)
        dup_within_norm = int(norm.duplicated().sum())
    else:
        dup_within_norm = 0

    uniq_stats = safe_quantiles(validated_scored2.get("uniqueness_score", pd.Series(dtype=float)))
    rank_stats = safe_quantiles(validated_scored2.get("rank_score", pd.Series(dtype=float)))

    summary_final = {
        **summary_early,
        "duplicates_within_output_normalized": int(dup_within_norm),
        "uniqueness_stats": uniq_stats,
        "rank_stats": rank_stats,
        "phase": "done",
    }

    template_df = load_output_template(cfg)
    final_df = build_final_from_template(validated_scored2, template_df)

    # Emit done (frontend MUST set loading=false on this)
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
