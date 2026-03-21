#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.taxonomy_retriever import L2Retriever


# ----------------------------
# L1 Wrapper (backward compatible)
# ----------------------------
@dataclass
class TaxonomyL1Bundle:
    """
    Wrapper that behaves like the original sklearn Pipeline for L1 prediction,
    but ALSO carries a learned "vertical fallback" L1 model.

    This keeps existing code working if it does:
        l1_model = joblib.load(...)
        l1_model.predict(...)
        l1_model.predict_proba(...)

    because we proxy those calls to `self.model`.
    """
    model: Pipeline

    # Optional learned fallback model trained only on non-B2B L1 labels
    vertical_model: Optional[Pipeline] = None
    b2b_label: str = "B2B Audience"

    # Runtime knobs (used later in pipeline/taxonomy_model)
    # If main L1 predicts B2B, and vertical model is confident enough, we can override.
    vertical_min_confidence: float = 0.55
    vertical_margin: float = 0.08  # optional margin vs "UNKNOWN" etc (kept for future)

    # --- proxy methods/attrs so old code keeps working ---
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("Underlying L1 model does not support predict_proba()")

    def decision_function(self, X):
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        raise AttributeError("Underlying L1 model does not support decision_function()")

    @property
    def classes_(self):
        return getattr(self.model, "classes_", None)


# ----------------------------
# Helpers
# ----------------------------

def clean(s: str) -> str:
    return str(s or "").strip()


def split_parts(name: str, sep: str) -> List[str]:
    """
    Robust split on '>' regardless of whitespace around it.
    """
    s = clean(name)
    if not s:
        return []
    parts = re.split(r"\s*>\s*", s)
    return [p.strip() for p in parts if p and p.strip()]


def extract_l1_l2(segment_name: str, sep: str, provider: str) -> Tuple[str, str]:
    """
    Expects:
      Cybba > L1 > L2 > Leaf
    Returns:
      (L1, L2) or ("", "") if not parseable.
    """
    parts = split_parts(segment_name, sep)

    if parts and parts[0].lower() == provider.lower():
        parts = parts[1:]

    # need at least L1, L2, Leaf
    if len(parts) < 3:
        return "", ""

    l1 = clean(parts[0])
    l2 = clean(parts[1])
    return l1, l2


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Stratify only if EVERY class has >= 2 samples.
    Otherwise, do a normal random split (prevents crash on tiny classes).
    """
    counts = Counter(y)
    min_count = min(counts.values()) if counts else 0

    if min_count < 2:
        logging.warning(
            "Not stratifying split because least populated class has %d sample(s). "
            "Falling back to non-stratified train/test split.",
            min_count,
        )
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def _pick_name_col(df: pd.DataFrame, override: str) -> str:
    if override:
        return override
    if "New Segment Name" in df.columns:
        return "New Segment Name"
    if "Segment Name" in df.columns:
        return "Segment Name"
    if "name" in df.columns:
        return "name"
    return "New Segment Name"


def _pick_desc_col(df: pd.DataFrame, override: str) -> str:
    if override:
        return override
    if "Segment Description" in df.columns:
        return "Segment Description"
    if "description" in df.columns:
        return "description"
    return "Segment Description"


def _build_text_series(df: pd.DataFrame, name_col: str, desc_col: str) -> pd.Series:
    """
    Builds feature text. If extra helpful columns exist, include them.
    """
    base = (df[name_col].astype(str) + " | " + df[desc_col].astype(str)).astype(str)

    extras: List[str] = []
    for c in ["Field Name", "Value Name", "LiveRamp Field Name", "LiveRamp Value Name"]:
        if c in df.columns:
            extras.append(c)

    if extras:
        extra_txt = df[extras].astype(str).agg(" | ".join, axis=1)
        base = (base + " | " + extra_txt).astype(str)

    return base


def _read_train_files(input_dir: Path, files: List[str]) -> List[Tuple[Path, pd.DataFrame]]:
    """
    Read each file independently so we can detect its format per-file.
    Supports .gz as well.
    """
    loaded: List[Tuple[Path, pd.DataFrame]] = []

    for f in files:
        f = clean(f)
        if not f:
            continue
        p = Path(f)
        if not p.is_absolute():
            p = input_dir / f
        if not p.exists():
            raise FileNotFoundError(f"Training file not found: {p}")
        logging.info("Loading training file: %s", p)

        compression = "gzip" if str(p).endswith(".gz") else None
        loaded.append((p, pd.read_csv(p, dtype=str, keep_default_na=False, compression=compression)))

    if not loaded:
        raise ValueError("No valid training files found in training.taxonomy_train_files.")
    return loaded


def _build_labels_from_df(
    df: pd.DataFrame,
    *,
    provider: str,
    sep: str,
    name_col: str,
    desc_col: str,
) -> pd.DataFrame:
    """
    Supports two formats:

    A) Prepared training format:
       - has columns: text, L1, L2 (optional Leaf)
       -> uses them directly (no parsing)

    B) Original format:
       - has name_col and desc_col
       - parses L1/L2 from name_col using extract_l1_l2()
    """
    # A) Prepared format
    if {"text", "L1", "L2"}.issubset(df.columns):
        if "Leaf" not in df.columns:
            logging.warning("Prepared file missing Leaf column; Leaf model will be skipped.")
        train_df = df.copy()
        train_df["text"] = train_df["text"].astype(str)
        train_df["L1"] = train_df["L1"].astype(str).map(clean)
        train_df["L2"] = train_df["L2"].astype(str).map(clean)

        if "Leaf" in train_df.columns:
            train_df["Leaf"] = train_df["Leaf"].astype(str).map(clean)

        if "Leaf" in train_df.columns:
            train_df = train_df[(train_df["L1"] != "") & (train_df["L2"] != "") & (train_df["Leaf"] != "")].copy()
        else:
            train_df = train_df[(train_df["L1"] != "") & (train_df["L2"] != "")].copy()

        train_df["__name_col_used__"] = "text"
        train_df["__desc_col_used__"] = ""
        return train_df

    # B) Original format
    missing = [c for c in [name_col, desc_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Training file missing required column(s): {missing}")

    l1_list: List[str] = []
    l2_list: List[str] = []
    keep_rows: List[int] = []

    for i, row in df.iterrows():
        l1, l2 = extract_l1_l2(row[name_col], sep=sep, provider=provider)
        if l1 and l2:
            l1_list.append(l1)
            l2_list.append(l2)
            keep_rows.append(i)

    train_df = df.loc[keep_rows].copy()
    train_df["L1"] = l1_list
    train_df["L2"] = l2_list
    train_df["__name_col_used__"] = name_col
    train_df["__desc_col_used__"] = desc_col

    train_df["text"] = _build_text_series(train_df, name_col=name_col, desc_col=desc_col)
    return train_df


def _tfidf_params_from_config(cfg: dict) -> Dict[str, Any]:
    """
    Reverted: do NOT force max_features=50k and do NOT force float32.
    If config provides values, use them; otherwise rely on sklearn defaults.
    """
    tfidf_cfg = cfg.get("model", {}).get("tfidf", {}) or {}

    params: Dict[str, Any] = {}
    if "ngram_min" in tfidf_cfg or "ngram_max" in tfidf_cfg:
        ngram_min = int(tfidf_cfg.get("ngram_min", 1))
        ngram_max = int(tfidf_cfg.get("ngram_max", 2))
        params["ngram_range"] = (ngram_min, ngram_max)

    if "min_df" in tfidf_cfg:
        params["min_df"] = int(tfidf_cfg.get("min_df", 1))

    if "stop_words" in tfidf_cfg:
        params["stop_words"] = tfidf_cfg.get("stop_words", "english")

    if "max_features" in tfidf_cfg:
        mf = tfidf_cfg.get("max_features", None)
        params["max_features"] = None if mf in ("", None) else int(mf)

    params["lowercase"] = True
    return params


def make_l1_model(cfg: dict) -> Pipeline:
    """
    LogisticRegression L1 classifier (original heavier approach).
    """
    tfidf_params = _tfidf_params_from_config(cfg)

    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LogisticRegression(
                max_iter=500,
                n_jobs=-1,
                solver="saga",
            )),
        ]
    )


def _fit_and_report(name: str, model: Pipeline, X_tr, X_te, y_tr, y_te) -> None:
    logging.info("Fitting %s (classes=%d, train=%d, test=%d)...",
                 name, len(set(y_tr)), len(y_tr), len(y_te))
    model.fit(X_tr, y_tr)

    if len(y_te) > 0:
        pred = model.predict(X_te)
        acc = accuracy_score(y_te, pred)
        logging.info("%s accuracy (holdout) = %.4f", name, acc)
        logging.info("%s report:\n%s", name, classification_report(y_te, pred, zero_division=0))


# ----------------------------
# Training
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "config.yml"),
        help="Path to YAML config file",
    )
    ap.add_argument(
        "--train-file",
        default="",
        help="Optional override: CSV file to train on (relative to cfg.paths.input_dir). "
             "If training.taxonomy_train_files is present in config, that list will be used unless you pass --train-file.",
    )
    ap.add_argument("--name-col", default="", help="Optional override name column.")
    ap.add_argument("--desc-col", default="", help="Optional override desc column.")
    ap.add_argument("--test-size", type=float, default=0.2)

    # Optional knobs for vertical fallback model
    ap.add_argument("--b2b-label", default="B2B Audience", help="Label name for B2B in your L1 labels.")
    ap.add_argument("--vertical-min-confidence", type=float, default=0.55, help="Saved into bundle for runtime use.")
    ap.add_argument("--vertical-margin", type=float, default=0.08, help="Saved into bundle for runtime use.")

    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    provider = clean(cfg["provider"]["cybba_name"]) or "Cybba"
    sep = clean(cfg.get("taxonomy", {}).get("separator", " > ")) or " > "

    input_dir = Path(cfg["paths"]["input_dir"])
    out_dir = Path(cfg.get("training", {}).get("model_dir", "Data/Models"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config_train_files = cfg.get("training", {}).get("taxonomy_train_files", None)

    if args.train_file:
        train_files = [args.train_file]
    elif isinstance(config_train_files, list) and len(config_train_files) > 0:
        train_files = config_train_files
    else:
        legacy = cfg.get("training", {}).get("taxonomy_train_file", "")
        train_files = [legacy] if legacy else []

    if not train_files or not any(clean(x) for x in train_files):
        raise ValueError("No training file(s) provided in config or --train-file.")

    loaded = _read_train_files(input_dir, train_files)

    labeled_parts: List[pd.DataFrame] = []
    total_raw = 0

    for p, df in loaded:
        total_raw += len(df)

        name_col = _pick_name_col(df, args.name_col)
        desc_col = _pick_desc_col(df, args.desc_col)

        logging.info("File: %s", p)
        logging.info("Using name_col=%r desc_col=%r (when applicable)", name_col, desc_col)

        labeled = _build_labels_from_df(df, provider=provider, sep=sep, name_col=name_col, desc_col=desc_col)
        logging.info("Usable labeled rows from this file: %d / %d", len(labeled), len(df))

        labeled_parts.append(labeled)

    train_df = pd.concat(labeled_parts, ignore_index=True)

    logging.info(
        "Total usable training rows (non-empty L1/L2%s): %d / %d (raw across files)",
        " + Leaf" if "Leaf" in train_df.columns else "",
        len(train_df),
        total_raw,
    )

    logging.info(
        "Training distribution: L1 classes=%d, L2 classes=%d%s",
        train_df["L1"].nunique(),
        train_df["L2"].nunique(),
        f", Leaf unique={train_df['Leaf'].nunique()}" if "Leaf" in train_df.columns else "",
    )

    X = train_df["text"].astype(str).tolist()

    # ----------------------------
    # L1 model (LogReg) - full label space
    # ----------------------------
    y1 = train_df["L1"].astype(str).tolist()
    l1_model = make_l1_model(cfg)

    X1_tr, X1_te, y1_tr, y1_te = safe_train_test_split(X, y1, test_size=args.test_size, random_state=42)
    _fit_and_report("L1", l1_model, X1_tr, X1_te, y1_tr, y1_te)

    # ----------------------------
    # NEW: Vertical fallback model (LogReg) - trained on non-B2B labels only
    # This is "smart" and label-driven (no keyword hardcoding).
    # ----------------------------
    b2b_label = clean(args.b2b_label) or "B2B Audience"

    vertical_model: Optional[Pipeline] = None
    non_b2b_mask = train_df["L1"].astype(str) != b2b_label
    non_b2b_df = train_df.loc[non_b2b_mask].copy()

    if len(non_b2b_df) < 200 or non_b2b_df["L1"].nunique() < 2:
        logging.warning(
            "Skipping vertical fallback model: not enough non-B2B data "
            "(rows=%d, unique_non_b2b_L1=%d).",
            len(non_b2b_df),
            non_b2b_df["L1"].nunique(),
        )
    else:
        Xv = non_b2b_df["text"].astype(str).tolist()
        yv = non_b2b_df["L1"].astype(str).tolist()

        vertical_model = make_l1_model(cfg)
        Xv_tr, Xv_te, yv_tr, yv_te = safe_train_test_split(
            Xv, yv, test_size=args.test_size, random_state=43
        )
        _fit_and_report("VerticalFallbackL1(non-B2B)", vertical_model, Xv_tr, Xv_te, yv_tr, yv_te)

        logging.info(
            "Vertical fallback trained on non-B2B labels: unique=%d",
            len(set(yv)),
        )

    # ----------------------------
    # L2 model (retrieval)
    # ----------------------------
    y2 = train_df["L2"].astype(str).tolist()
    logging.info("Building L2 retriever index (unique L2=%d)...", len(set(y2)))

    tfidf_params = _tfidf_params_from_config(cfg)
    l2_model = L2Retriever.fit_from_text(tfidf_params, X, y2)

    # ----------------------------
    # Leaf model (retrieval)
    # ----------------------------
    if "Leaf" in train_df.columns:
        y3 = train_df["Leaf"].astype(str).tolist()
        logging.info("Building Leaf retriever index (unique Leaf=%d)...", len(set(y3)))
        leaf_model = L2Retriever.fit_from_text(tfidf_params, X, y3)
    else:
        leaf_model = None
        logging.warning("No Leaf column found; skipping Leaf retriever training.")

    # Save artifacts
    l1_path = out_dir / "cybba_taxonomy_L1.joblib"
    l2_path = out_dir / "cybba_taxonomy_L2.joblib"
    leaf_path = out_dir / "cybba_taxonomy_Leaf.joblib"

    # ✅ Save a backward-compatible wrapper at the SAME path
    l1_bundle = TaxonomyL1Bundle(
        model=l1_model,
        vertical_model=vertical_model,
        b2b_label=b2b_label,
        vertical_min_confidence=float(args.vertical_min_confidence),
        vertical_margin=float(args.vertical_margin),
    )

    joblib.dump(l1_bundle, l1_path)
    joblib.dump(l2_model, l2_path)
    logging.info("Saved: %s", l1_path)
    logging.info("Saved: %s", l2_path)

    if leaf_model is not None:
        joblib.dump(leaf_model, leaf_path)
        logging.info("Saved: %s", leaf_path)

    logging.info(
        "Done. L1 wrapper saved with vertical_model=%s (b2b_label=%r, min_conf=%.2f).",
        bool(vertical_model),
        b2b_label,
        float(args.vertical_min_confidence),
    )


if __name__ == "__main__":
    main()