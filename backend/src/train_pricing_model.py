# backend/src/train_pricing_model.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pricing_model import PRICE_COLS, PricingModelBundle  # noqa: E402
# NOTE: PricingModelBundle should include:
#   tier_quantiles: Dict[Tuple[str,str], Dict[str, Tuple[float,float]]]
#   global_quantiles: Dict[str, Tuple[float,float]]
#   calibration_params: Dict[str, Tuple[float, float]]   # (a,b) where true ~= a + b*pred


TEXT_COLS = ["Segment Name", "Segment Description", "Components"]
CAT_COLS = ["Provider Name", "Country", "Currency", "Audience Type", "Taxonomy L1", "Taxonomy L2"]
NUM_COLS = ["Estimated Cookie Reach", "Estimated iOS Reach", "Android Reach"]

TRAIN_PROVIDER = "Cybba"
TRAIN_COUNTRY = "USA"
TRAIN_CURRENCY = "USD"


# ----------------------------
# Helpers
# ----------------------------
def _clean_text_series(s: pd.Series) -> pd.Series:
    s2 = s.fillna("").astype(str)
    s2 = s2.map(lambda x: re.sub(r"\s+", " ", x).strip())
    s2 = s2.replace({"nan": "", "None": "", "NULL": "", "NaN": ""})
    return s2


def _split_taxonomy(name: str) -> Tuple[str, str, str]:
    parts = [p.strip() for p in str(name or "").split(">") if p.strip()]
    if parts and parts[0].lower() == "cybba":
        parts = parts[1:]
    aud = parts[0] if len(parts) >= 1 else "UNKNOWN"
    l1 = parts[1] if len(parts) >= 2 else "UNKNOWN"
    l2 = parts[2] if len(parts) >= 3 else "UNKNOWN"
    return aud or "UNKNOWN", l1 or "UNKNOWN", l2 or "UNKNOWN"


def _to_numeric_price_series(s: pd.Series) -> pd.Series:
    x = s.copy()
    x = x.fillna("").astype(str).str.strip()
    x = x.str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    x = x.str.replace("%", "", regex=False)
    x = x.replace({"": np.nan, "—": np.nan, "-": np.nan, "None": np.nan, "nan": np.nan})
    return pd.to_numeric(x, errors="coerce")


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
        "Components",
    ] + list(PRICE_COLS)

    for c in keep:
        if c not in df.columns:
            df[c] = np.nan

    out = df[keep].copy()

    out["Provider Name"] = TRAIN_PROVIDER
    out["Country"] = TRAIN_COUNTRY
    out["Currency"] = TRAIN_CURRENCY

    for c in TEXT_COLS:
        out[c] = _clean_text_series(out[c])

    for c in NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    parsed = out["Segment Name"].map(_split_taxonomy)
    out["Audience Type"] = parsed.map(lambda t: t[0])
    out["Taxonomy L1"] = parsed.map(lambda t: t[1])
    out["Taxonomy L2"] = parsed.map(lambda t: t[2])

    return out


def _build_baseline_table(df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
    gcols = ["Audience Type", "Taxonomy L1"]
    tmp = df[gcols + list(PRICE_COLS)].copy()

    for c in PRICE_COLS:
        tmp[c] = _to_numeric_price_series(tmp[c])

    med = tmp.groupby(gcols, dropna=False)[list(PRICE_COLS)].median(numeric_only=True)

    base: Dict[Tuple[str, str], Dict[str, float]] = {}
    for idx, row in med.iterrows():
        aud, l1 = idx
        key = (str(aud), str(l1))
        base[key] = {c: float(row[c]) if pd.notna(row[c]) else float("nan") for c in PRICE_COLS}
    return base


def _build_quantile_tables(
    df: pd.DataFrame,
    *,
    q_lo: float = 0.05,
    q_hi: float = 0.95,
    min_rows_per_tier: int = 200,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Tuple[float, float]]], Dict[str, Tuple[float, float]]]:
    gcols = ["Audience Type", "Taxonomy L1"]
    tmp = df[gcols + list(PRICE_COLS)].copy()
    for c in PRICE_COLS:
        tmp[c] = _to_numeric_price_series(tmp[c])

    global_q: Dict[str, Tuple[float, float]] = {}
    for c in PRICE_COLS:
        s = tmp[c].dropna()
        if len(s) == 0:
            continue
        lo = float(s.quantile(q_lo))
        hi = float(s.quantile(q_hi))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            global_q[c] = (lo, hi)

    tier_q: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    for (aud, l1), g in tmp.groupby(gcols, dropna=False):
        key = (str(aud), str(l1))
        d: Dict[str, Tuple[float, float]] = {}
        for c in PRICE_COLS:
            s = g[c].dropna()
            if len(s) < min_rows_per_tier:
                continue
            lo = float(s.quantile(q_lo))
            hi = float(s.quantile(q_hi))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                d[c] = (lo, hi)
        if d:
            tier_q[key] = d

    return tier_q, global_q


def _make_knn_text(df: pd.DataFrame) -> pd.Series:
    name = df["Segment Name"].fillna("").astype(str)
    desc = df["Segment Description"].fillna("").astype(str)
    comps = df["Components"].fillna("").astype(str)
    return (name + " | " + desc + " | " + comps).str.strip()


def _choose_knn_training_subset(
    df: pd.DataFrame,
    max_per_tier: int,
    max_total: int,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    tiers = df.groupby(["Audience Type", "Taxonomy L1"], dropna=False)

    chunks = []
    for _, g in tiers:
        if len(g) <= max_per_tier:
            chunks.append(g)
        else:
            chunks.append(g.sample(n=max_per_tier, random_state=int(rng.randint(0, 2**31 - 1))))

    out = pd.concat(chunks, ignore_index=True)
    if len(out) > max_total:
        out = out.sample(n=max_total, random_state=seed).reset_index(drop=True)

    return out.reset_index(drop=True)


def _detect_name_col(df: pd.DataFrame) -> str:
    for c in ["Segment Name", "New Segment Name", "Proposed New Segment Name"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find a segment name column in Cybba pricing CSV.")


def _fit_global_calibration(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    *,
    min_pairs: int = 40,
    clip_slope: Tuple[float, float] = (0.2, 3.0),
    clip_intercept: Tuple[float, float] = (-10.0, 10.0),
) -> Dict[str, Tuple[float, float]]:
    """
    Fit per-column linear calibration:
        truth ≈ a + b * pred

    Uses robust trimming (winsorize) to avoid a few outliers dominating.
    """
    params: Dict[str, Tuple[float, float]] = {}

    for c in PRICE_COLS:
        if c not in pred_df.columns or c not in truth_df.columns:
            continue

        y = pd.to_numeric(truth_df[c], errors="coerce")
        x = pd.to_numeric(pred_df[c], errors="coerce")
        m = x.notna() & y.notna()
        if int(m.sum()) < min_pairs:
            continue

        xx = x[m].astype(float).to_numpy()
        yy = y[m].astype(float).to_numpy()

        # trim extremes (winsorize at 2%/98%)
        x_lo, x_hi = np.quantile(xx, [0.02, 0.98])
        y_lo, y_hi = np.quantile(yy, [0.02, 0.98])
        xx = np.clip(xx, x_lo, x_hi)
        yy = np.clip(yy, y_lo, y_hi)

        # least squares fit: yy = a + b*xx
        A = np.vstack([np.ones_like(xx), xx]).T
        sol, _, _, _ = np.linalg.lstsq(A, yy, rcond=None)
        a = float(sol[0])
        b = float(sol[1])

        b = float(np.clip(b, clip_slope[0], clip_slope[1]))
        a = float(np.clip(a, clip_intercept[0], clip_intercept[1]))

        params[c] = (a, b)

    return params


def train(
    csv_path: Path,
    out_path: Path,
    *,
    # kNN index sizing knobs
    knn_max_per_tier: int = 20000,
    knn_max_total: int = 200000,
    knn_max_features: int = 200000,
    # quantile guardrails knobs
    q_lo: float = 0.05,
    q_hi: float = 0.95,
    quantile_min_rows_per_tier: int = 200,
    # optional Cybba calibration data
    cybba_prices_csv: Optional[Path] = None,
) -> None:
    df = pd.read_csv(csv_path, low_memory=False)
    df = _prep_training_df(df)

    prices_numeric = df[list(PRICE_COLS)].copy()
    for c in PRICE_COLS:
        prices_numeric[c] = _to_numeric_price_series(prices_numeric[c])

    fallback_means: Dict[str, float] = {}
    for c in PRICE_COLS:
        s = prices_numeric[c].dropna()
        fallback_means[c] = float(s.mean()) if len(s) else 0.0

    baseline_table = _build_baseline_table(df)

    tier_quantiles, global_quantiles = _build_quantile_tables(
        df,
        q_lo=q_lo,
        q_hi=q_hi,
        min_rows_per_tier=quantile_min_rows_per_tier,
    )

    usable_mask = prices_numeric.notna().any(axis=1)
    usable_df = df.loc[usable_mask].copy().reset_index(drop=True)
    usable_prices = prices_numeric.loc[usable_mask].copy().reset_index(drop=True)

    print(f"[train_pricing_model] total rows: {len(df)}")
    print(f"[train_pricing_model] usable rows for kNN (has any price): {len(usable_df)}")

    joined = usable_df.copy()
    for c in PRICE_COLS:
        joined[c] = usable_prices[c].values

    knn_df = _choose_knn_training_subset(
        joined,
        max_per_tier=int(knn_max_per_tier),
        max_total=int(knn_max_total),
        seed=42,
    )

    knn_prices = knn_df[list(PRICE_COLS)].copy()
    for c in PRICE_COLS:
        knn_prices[c] = pd.to_numeric(knn_prices[c], errors="coerce")

    corpus = _make_knn_text(knn_df)
    if (corpus.str.len() == 0).mean() > 0.95:
        raise ValueError("kNN training corpus is ~empty. Check Segment Name/Description/Components content.")

    vectorizer = TfidfVectorizer(
        max_features=int(knn_max_features),
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
        min_df=2,
        token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9]+\b",
    )

    print(f"[train_pricing_model] fitting TF-IDF on {len(corpus)} rows (max_features={knn_max_features})...")
    X = vectorizer.fit_transform(corpus)

    knn_aud = knn_df["Audience Type"].fillna("UNKNOWN").astype(str).tolist()
    knn_l1 = knn_df["Taxonomy L1"].fillna("UNKNOWN").astype(str).tolist()

    # ----------------------------
    # Optional: Global calibration using Cybba prices CSV
    # ----------------------------
    calibration_params: Dict[str, Tuple[float, float]] = {}
    if cybba_prices_csv is not None:
        if not cybba_prices_csv.exists():
            raise FileNotFoundError(f"cybba_prices_csv not found: {cybba_prices_csv}")

        cyb = pd.read_csv(cybba_prices_csv, low_memory=False)
        name_col = _detect_name_col(cyb)

        # build minimal df in training schema for pricing_model.predict_prices()
        cyb_tmp = pd.DataFrame(index=cyb.index)
        cyb_tmp["Proposed New Segment Name"] = cyb[name_col].fillna("").astype(str)
        if "Segment Description" in cyb.columns:
            cyb_tmp["Segment Description"] = cyb["Segment Description"]
        if "Non Derived Segments utilized" in cyb.columns:
            cyb_tmp["Non Derived Segments utilized"] = cyb["Non Derived Segments utilized"]

        # true prices
        truth = pd.DataFrame(index=cyb.index)
        for c in PRICE_COLS:
            if c in cyb.columns:
                truth[c] = _to_numeric_price_series(cyb[c])
            else:
                truth[c] = np.nan

        # IMPORTANT:
        # We need predictions from the *same bundle structure* that pricing_model.py expects.
        # We can construct a temporary bundle (without calibration) and call pricing_model.predict_prices()
        # BUT we can't import predict_prices here without circular behavior in some setups.
        #
        # So: we fit calibration in a *later pass* (recommended) OR you can add:
        #   from pricing_model import predict_prices as predict_prices_fn
        # and call it here.
        #
        # If your repo is fine with that import, uncomment next two lines:

        from pricing_model import predict_prices as predict_prices_fn  # type: ignore

        tmp_bundle = PricingModelBundle(
            price_cols=list(PRICE_COLS),
            fallback_means=fallback_means,
            baseline_table=baseline_table,
            tier_quantiles=tier_quantiles,
            global_quantiles=global_quantiles,
            calibration_params={},  # none yet

            knn_vectorizer=vectorizer,
            knn_matrix=X,
            knn_prices=knn_prices.astype(float),
            knn_audience=knn_aud,
            knn_tax_l1=knn_l1,
        )

        pred = predict_prices_fn(tmp_bundle, cyb_tmp)

        calibration_params = _fit_global_calibration(pred, truth, min_pairs=40)
        print(f"[train_pricing_model] calibration learned cols: {sorted(calibration_params.keys())}")

    bundle = PricingModelBundle(
        price_cols=list(PRICE_COLS),
        fallback_means=fallback_means,
        baseline_table=baseline_table,

        tier_quantiles=tier_quantiles,
        global_quantiles=global_quantiles,

        calibration_params=calibration_params,  # ✅ NEW

        knn_vectorizer=vectorizer,
        knn_matrix=X,
        knn_prices=knn_prices.astype(float),
        knn_audience=knn_aud,
        knn_tax_l1=knn_l1,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)

    print(f"[train_pricing_model] Saved pricing model bundle to: {out_path}")
    print(f"[train_pricing_model] baseline tiers: {len(baseline_table)}")
    print(f"[train_pricing_model] quantile tiers: {len(tier_quantiles)} | global quantiles: {len(global_quantiles)}")
    print(f"[train_pricing_model] knn rows: {len(knn_df)} | knn nnz: {int(X.nnz)}")
    if calibration_params:
        for c, (a, b) in calibration_params.items():
            print(f"[train_pricing_model] calib[{c}]: a={a:.4f}, b={b:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--knn_max_per_tier", type=int, default=20000)
    ap.add_argument("--knn_max_total", type=int, default=200000)
    ap.add_argument("--knn_max_features", type=int, default=200000)

    ap.add_argument("--q_lo", type=float, default=0.05)
    ap.add_argument("--q_hi", type=float, default=0.95)
    ap.add_argument("--quantile_min_rows_per_tier", type=int, default=200)

    # ✅ new: Cybba calibration CSV (optional)
    ap.add_argument("--cybba_prices_csv", type=str, default="")

    args = ap.parse_args()

    cybba_csv = Path(args.cybba_prices_csv) if args.cybba_prices_csv else None

    train(
        Path(args.csv),
        Path(args.out),
        knn_max_per_tier=int(args.knn_max_per_tier),
        knn_max_total=int(args.knn_max_total),
        knn_max_features=int(args.knn_max_features),
        q_lo=float(args.q_lo),
        q_hi=float(args.q_hi),
        quantile_min_rows_per_tier=int(args.quantile_min_rows_per_tier),
        cybba_prices_csv=cybba_csv,
    )
