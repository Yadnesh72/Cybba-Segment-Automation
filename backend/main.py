#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# -------------------------------------------------
# Make backend/src importable
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from segment_expansion_model import generate_proposals
from validators import validate_and_prepare


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def load_config() -> dict:
    cfg_path = BASE_DIR / "config" / "config.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def price_segments_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporary pricing stub.
    Replace with pricing_engine.price_segments later.
    """
    out = df.copy()
    out["Digital Ad Targeting Price (CPM)"] = 10.00
    out["Content Marketing Price (CPM)"] = 0.00
    out["TV Targeting Price (CPM)"] = 0.00
    out["Cost Per Click"] = 0.00
    out["Programmatic % of Media"] = 0.00
    out["CPM Cap"] = 10.00
    out["Advertiser Direct % of Media"] = 0.00
    return out


def build_final_output_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporary formatter stub.
    Replace with reading Output_format.csv later.
    """
    final_cols = [
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

    required_in = ["Proposed New Segment Name", "Non Derived Segments utilized"] + final_cols[3:]
    missing = [c for c in required_in if c not in df.columns]
    if missing:
        raise ValueError(f"build_final_output_v1 missing required columns in input df: {missing}")

    out = pd.DataFrame()
    out["New Segment Name"] = df["Proposed New Segment Name"]
    out["Non Derived Segments utilized"] = df["Non Derived Segments utilized"]
    out["Segment Description"] = ""

    for c in final_cols[3:]:
        out[c] = df[c]

    return out[final_cols]


def safe_outfile(cfg: dict, key: str, default_name: str) -> str:
    return str(cfg.get("output", {}).get(key, default_name))


def write_df(df: pd.DataFrame, path: Path, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote %s (rows=%d)", path, len(df))


def empty_proposals_df() -> pd.DataFrame:
    """
    Ensures proposals/validated output always has the expected columns,
    even when there are 0 rows.
    """
    cols = [
        "Competitor Provider",
        "Competitor Segment Name",
        "Competitor Segment ID",
        "Proposed New Segment Name",
        "Non Derived Segments utilized",
        "Composition Similarity",
        "Closest Cybba Segment",
        "Closest Cybba Similarity",
        "Taxonomy",
    ]
    return pd.DataFrame(columns=cols)


def empty_final_df() -> pd.DataFrame:
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


def empty_coverage_df() -> pd.DataFrame:
    cols = [
        "Competitor Provider",
        "Competitor Segment Name",
        "Competitor Segment ID",
        "Closest Cybba Segment",
        "Similarity",
    ]
    return pd.DataFrame(columns=cols)


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------

def main() -> None:
    cfg = load_config()

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("pipeline")
    logger.info("Loaded config. Output dir: %s", out_dir)

    # Output file names (configurable)
    proposals_name = safe_outfile(cfg, "proposals_filename", "proposals.csv")
    validated_name = safe_outfile(cfg, "proposals_validated_filename", "proposals_validated.csv")
    final_name = safe_outfile(cfg, "final_filename", "Cybba_New_Additional_Segments.csv")
    coverage_name = safe_outfile(cfg, "coverage_filename", "coverage.csv")

    proposals_path = out_dir / proposals_name
    validated_path = out_dir / validated_name
    final_path = out_dir / final_name
    coverage_path = out_dir / coverage_name

    # 1) MODEL
    proposals_df, coverage_df = generate_proposals(cfg)
    logger.info("Model output: proposals=%d coverage=%d", len(proposals_df), len(coverage_df))

    # Always write raw outputs (schema-safe)
    if proposals_df is None or proposals_df.empty:
        write_df(empty_proposals_df(), proposals_path, logger)
    else:
        write_df(proposals_df, proposals_path, logger)

    if coverage_df is None or coverage_df.empty:
        write_df(empty_coverage_df(), coverage_path, logger)
    else:
        write_df(coverage_df, coverage_path, logger)

    if proposals_df is None or proposals_df.empty:
        logger.warning("Model produced 0 proposals. Skipping validation/pricing/final output.")
        write_df(empty_proposals_df(), validated_path, logger)
        write_df(empty_final_df(), final_path, logger)
        return

    # 2) LOAD INPUTS FOR VALIDATORS
    input_dir = Path(cfg["paths"]["input_dir"])
    A = pd.read_csv(input_dir / cfg["files"]["input_a"], dtype=str, keep_default_na=False)
    B = pd.read_csv(input_dir / cfg["files"]["input_b"], dtype=str, keep_default_na=False)

    # 3) VALIDATION
    validated_df, report = validate_and_prepare(
        proposals_df,
        underived_df=B,
        distributed_df=A,
        cfg=cfg,
        logger=logging.getLogger("validators"),
    )
    logger.info("Validation report: %s", report)

    # Sort best first
    if not validated_df.empty and "Composition Similarity" in validated_df.columns:
        validated_df = validated_df.sort_values("Composition Similarity", ascending=False)

    # Always write validated proposals (schema-safe)
    if validated_df.empty:
        write_df(empty_proposals_df(), validated_path, logger)
    else:
        write_df(validated_df, validated_path, logger)

    # 4) PRICING (stub)
    priced_df = price_segments_v1(validated_df)

    # 5) FINAL FORMATTING (stub)
    if priced_df.empty:
        final_df = empty_final_df()
    else:
        final_df = build_final_output_v1(priced_df)

    # 6) WRITE FINAL (NO head(100) — let model decide row count)
    write_df(final_df, final_path, logger)


if __name__ == "__main__":
    main()
