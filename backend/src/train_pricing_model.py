# backend/src/train_pricing_model.py
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor

import joblib

PRICE_COLS = [
    "Digital Ad Targeting Price (CPM)",
    "Content Marketing Price (CPM)",
    "TV Targeting Price (CPM)",
    "Cost Per Click",
    "Programmatic % of Media",
    "CPM Cap",
    "Advertiser Direct % of Media",
]

TEXT_COLS = ["Segment Name", "Segment Description", "Components"]
CAT_COLS = ["Provider Name", "Country", "Currency"]
NUM_COLS = ["Estimated Cookie Reach", "Estimated iOS Reach", "Android Reach"]


def _clean_text_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.map(lambda x: re.sub(r"\s+", " ", x).strip())
    return s


def _prep_training_df(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "Provider Name",
        "Country",
        "Currency",
        "Segment Name",
        "Segment Description",
        "Estimated Cookie Reach",
        "Estimated iOS Reach",
        "Android Reach",
    ] + PRICE_COLS

    for c in keep:
        if c not in df.columns:
            df[c] = np.nan

    out = df[keep].copy()

    # Optional feature: composition
    if "Components" not in out.columns:
        out["Components"] = ""
    out["Components"] = out.get("Components", "")

    # Text
    for c in TEXT_COLS:
        out[c] = _clean_text_series(out[c])

    # Categorical
    for c in CAT_COLS:
        out[c] = out[c].fillna("").astype(str)

    # Numeric features
    for c in NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def _text_has_signal(s: pd.Series) -> bool:
    s = _clean_text_series(s)

    nonempty = s.map(lambda x: len(x) > 0).sum()
    if nonempty < 10:
        return False

    alpha_rows = s.map(lambda x: bool(re.search(r"[A-Za-z]", x))).sum()
    if alpha_rows < 10:
        return False

    return True


def _to_numeric_price_series(s: pd.Series) -> pd.Series:
    """
    Robust parsing for price-like values:
      - removes $, commas, whitespace
      - converts "12%" -> 12 (numeric)
      - handles empty strings
    """
    if s is None:
        return pd.Series(dtype=float)

    x = s.copy()

    # If it's already numeric-ish, just coerce
    if not pd.api.types.is_object_dtype(x) and not pd.api.types.is_string_dtype(x):
        return pd.to_numeric(x, errors="coerce")

    x = x.fillna("").astype(str).str.strip()

    # Remove $ and commas
    x = x.str.replace("$", "", regex=False).str.replace(",", "", regex=False)

    # Convert percent strings "12%" -> "12"
    x = x.str.replace("%", "", regex=False)

    # Empty -> NaN
    x = x.replace({"": np.nan, "—": np.nan, "-": np.nan, "None": np.nan, "nan": np.nan})

    return pd.to_numeric(x, errors="coerce")


def train(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path, low_memory=False)
    df = _prep_training_df(df)

    # Build X
    X = df[TEXT_COLS + CAT_COLS + NUM_COLS].copy()

    # Build y with strong cleaning
    y = df[PRICE_COLS].copy()
    for c in PRICE_COLS:
        y[c] = _to_numeric_price_series(y[c])

    # Keep rows with at least one target (initial filter)
    any_target = y.notna().any(axis=1)
    X = X.loc[any_target].reset_index(drop=True)
    y = y.loc[any_target].reset_index(drop=True)

    if len(X) == 0:
        raise ValueError(
            "No training rows with any pricing targets after parsing. "
            "Verify your CSV includes the price columns and they contain numeric values."
        )

    # IMPORTANT: Ridge/MultiOutput cannot handle NaNs in y.
    # Phase 1: require all targets present.
    all_targets = y.notna().all(axis=1)
    before = len(X)
    X = X.loc[all_targets].reset_index(drop=True)
    y = y.loc[all_targets].reset_index(drop=True)
    after = len(X)

    print(f"[train_pricing_model] rows with ANY target: {before}")
    print(f"[train_pricing_model] rows with ALL targets (used): {after}")

    if after < 1000:
        raise ValueError(
            f"Too few rows after requiring all targets present ({after}). "
            "Your dataset may not have complete pricing for every column. "
            "If you want, we can switch to training per-target models using only rows where that target exists."
        )

    # Determine usable text columns
    usable_text_cols = [c for c in TEXT_COLS if _text_has_signal(X[c])]
    if not usable_text_cols:
        raise ValueError(
            "All text columns look empty/low-signal in this dataset sample, "
            "so TF-IDF would have an empty vocabulary."
        )

    print(f"[train_pricing_model] usable text cols: {usable_text_cols}")

    # TF-IDF per usable text column
    text_transformers = []
    for tc in usable_text_cols:
        text_transformers.append(
            (
                f"tfidf_{tc}",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    lowercase=True,
                    stop_words=None,
                    min_df=2,
                    token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9]+\b",
                ),
                tc,
            )
        )

    pre = ColumnTransformer(
        transformers=[
            *text_transformers,
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]), NUM_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    base = Ridge(alpha=2.0, random_state=42)
    model = Pipeline(
        steps=[
            ("pre", pre),
            ("reg", MultiOutputRegressor(base)),
        ]
    )

    model.fit(X, y)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Saved pricing model to: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to LiveRamp marketplace CSV")
    ap.add_argument("--out", required=True, help="Where to save the trained model (.joblib)")
    args = ap.parse_args()

    train(Path(args.csv), Path(args.out))
