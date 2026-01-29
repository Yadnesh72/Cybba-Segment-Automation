#!/usr/bin/env python3


from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


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


def make_model():
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=1)),
        ]
    )


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
        help="Optional override: CSV file to train on (relative to cfg.paths.input_dir).",
    )
    ap.add_argument(
        "--name-col",
        default="",
        help="Optional override: name column. If not set, auto-picks 'New Segment Name' then 'Segment Name'.",
    )
    ap.add_argument(
        "--desc-col",
        default="",
        help="Optional override: description column. If not set, defaults to 'Segment Description'.",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size fraction (only used when split is possible).",
    )
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    provider = clean(cfg["provider"]["cybba_name"]) or "Cybba"
    sep = clean(cfg.get("taxonomy", {}).get("separator", " > ")) or " > "

    # Which file to train on
    train_file = args.train_file or cfg.get("training", {}).get("taxonomy_train_file", "")
    if not train_file:
        raise ValueError(
            "No training file provided.\n"
            "Either pass --train-file <file.csv> or set in config:\n"
            "training:\n  taxonomy_train_file: \"Output_format.csv\""
        )

    input_dir = Path(cfg["paths"]["input_dir"])
    train_path = input_dir / train_file

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    out_dir = Path(cfg.get("training", {}).get("model_dir", "Data/Models"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Training data: %s", train_path)

    df = pd.read_csv(train_path, dtype=str, keep_default_na=False)

    # Pick columns
    desc_col = args.desc_col or "Segment Description"

    # Auto-pick name column if not provided
    if args.name_col:
        name_col = args.name_col
    else:
        if "New Segment Name" in df.columns:
            name_col = "New Segment Name"
        elif "Segment Name" in df.columns:
            name_col = "Segment Name"
        else:
            name_col = "New Segment Name"  # used for error message

    logging.info("Using name_col=%r desc_col=%r", name_col, desc_col)

    # Validate columns
    missing = [c for c in [name_col, desc_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Training file missing required column(s): {missing}")

    # Build labels
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

    logging.info("Usable training rows (parseable L1/L2): %d / %d", len(train_df), len(df))

    if len(train_df) < 20:
        raise ValueError(
            f"Too few parseable training rows ({len(train_df)}). "
            f"Your names must look like: {provider} > L1 > L2 > Leaf"
        )

    if len(train_df) < 200:
        logging.warning("Training set is small (%d). Model may be weak.", len(train_df))

    # Features
    X = (train_df[name_col].astype(str) + " | " + train_df[desc_col].astype(str)).tolist()

    # ----------------------------
    # L1 model
    # ----------------------------
    y1 = train_df["L1"].astype(str).tolist()

    l1_model = make_model()
    X1_tr, X1_te, y1_tr, y1_te = safe_train_test_split(X, y1, test_size=args.test_size, random_state=42)

    logging.info("Fitting L1 model (classes=%d, train=%d, test=%d)...",
                 len(set(y1)), len(y1_tr), len(y1_te))
    l1_model.fit(X1_tr, y1_tr)

    if len(y1_te) > 0:
        y1_pred = l1_model.predict(X1_te)
        acc1 = accuracy_score(y1_te, y1_pred)
        logging.info("L1 accuracy (holdout) = %.4f", acc1)
        # Keep report short; comment out if too noisy
        logging.info("L1 report:\n%s", classification_report(y1_te, y1_pred, zero_division=0))

    # ----------------------------
    # L2 model
    # ----------------------------
    y2 = train_df["L2"].astype(str).tolist()

    l2_model = make_model()
    X2_tr, X2_te, y2_tr, y2_te = safe_train_test_split(X, y2, test_size=args.test_size, random_state=42)

    logging.info("Fitting L2 model (classes=%d, train=%d, test=%d)...",
                 len(set(y2)), len(y2_tr), len(y2_te))
    l2_model.fit(X2_tr, y2_tr)

    if len(y2_te) > 0:
        y2_pred = l2_model.predict(X2_te)
        acc2 = accuracy_score(y2_te, y2_pred)
        logging.info("L2 accuracy (holdout) = %.4f", acc2)
        logging.info("L2 report:\n%s", classification_report(y2_te, y2_pred, zero_division=0))

    # Save artifacts
    l1_path = out_dir / "cybba_taxonomy_L1.joblib"
    l2_path = out_dir / "cybba_taxonomy_L2.joblib"

    joblib.dump(l1_model, l1_path)
    joblib.dump(l2_model, l2_path)

    logging.info("Saved: %s", l1_path)
    logging.info("Saved: %s", l2_path)
    logging.info("Done.")


if __name__ == "__main__":
    main()
