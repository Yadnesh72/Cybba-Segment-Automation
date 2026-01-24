# backend/src/pricing_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


@dataclass
class PricingDefaults:
    provider_name: str = "Cybba"
    country: str = "USA"
    currency: str = "USD"


def _round_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process predictions into clean marketplace-like outputs.
    - CPM fields and CPM Cap: 1 decimal, clipped >= 0
    - CPC: 2 decimals, clipped >= 0
    - % fields: integers 0..100
    """
    out = df.copy()

    # CPMs / cap -> 1 decimal
    for c in [
        "Digital Ad Targeting Price (CPM)",
        "Content Marketing Price (CPM)",
        "TV Targeting Price (CPM)",
        "CPM Cap",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
            out[c] = out[c].clip(lower=0.0).round(1)

    # CPC -> 2 decimals
    if "Cost Per Click" in out.columns:
        out["Cost Per Click"] = pd.to_numeric(out["Cost Per Click"], errors="coerce").fillna(0.0)
        out["Cost Per Click"] = out["Cost Per Click"].clip(lower=0.0).round(2)

    # % fields -> whole numbers 0..100
    for c in ["Programmatic % of Media", "Advertiser Direct % of Media"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
            out[c] = out[c].clip(lower=0.0, upper=100.0).round(0).astype(int)

    return out


def load_pricing_model(model_path: Path):
    """
    Loads the trained sklearn pipeline saved via joblib.
    """
    if joblib is None:
        raise RuntimeError("joblib not installed. Add joblib + scikit-learn to requirements.")
    if not model_path.exists():
        raise FileNotFoundError(f"Pricing model not found at: {model_path}")
    return joblib.load(model_path)


def predict_prices(
    model,
    segments_df: pd.DataFrame,
    defaults: Optional[PricingDefaults] = None,
) -> pd.DataFrame:
    """
    Build the feature frame expected by train_pricing_model.py and predict PRICE_COLS.

    segments_df accepted columns:
      - Proposed New Segment Name (preferred) OR New Segment Name
      - Segment Description (optional but recommended)
      - Non Derived Segments utilized (optional)

    Optional numeric inputs if available:
      - Estimated Cookie Reach
      - Estimated iOS Reach
      - Android Reach

    Returns:
      DataFrame aligned to segments_df.index with PRICE_COLS.
    """
    defaults = defaults or PricingDefaults()

    # Figure out which schema we have
    if "Proposed New Segment Name" in segments_df.columns:
        name_col = "Proposed New Segment Name"
    elif "New Segment Name" in segments_df.columns:
        name_col = "New Segment Name"
    else:
        # fail fast with a helpful error
        raise ValueError(
            "predict_prices expected 'Proposed New Segment Name' or 'New Segment Name' in segments_df."
        )

    desc_col = "Segment Description" if "Segment Description" in segments_df.columns else None
    comps_col = "Non Derived Segments utilized" if "Non Derived Segments utilized" in segments_df.columns else None

    # IMPORTANT: fillna('') before astype(str) to avoid "nan" strings + ensure no None
    seg_name = segments_df[name_col].fillna("").astype(str)
    seg_desc = segments_df[desc_col].fillna("").astype(str) if desc_col else pd.Series([""] * len(segments_df), index=segments_df.index)
    seg_comps = segments_df[comps_col].fillna("").astype(str) if comps_col else pd.Series([""] * len(segments_df), index=segments_df.index)

    # Build X exactly like training expects (names must match training script)
    X = pd.DataFrame(
        {
            "Provider Name": defaults.provider_name,
            "Country": defaults.country,
            "Currency": defaults.currency,
            "Segment Name": seg_name,
            "Segment Description": seg_desc,
            "Components": seg_comps,
            # optional numeric signals if present
            "Estimated Cookie Reach": pd.to_numeric(
                segments_df.get("Estimated Cookie Reach", np.nan), errors="coerce"
            ),
            "Estimated iOS Reach": pd.to_numeric(
                segments_df.get("Estimated iOS Reach", np.nan), errors="coerce"
            ),
            "Android Reach": pd.to_numeric(
                segments_df.get("Android Reach", np.nan), errors="coerce"
            ),
        },
        index=segments_df.index,
    )

    preds = model.predict(X)

    # MultiOutputRegressor returns shape (n, len(PRICE_COLS))
    preds_df = pd.DataFrame(preds, columns=PRICE_COLS, index=segments_df.index)

    # clean rounding / clipping
    preds_df = _round_outputs(preds_df)

    return preds_df
