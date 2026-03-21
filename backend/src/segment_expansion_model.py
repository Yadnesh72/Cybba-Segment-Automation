#!/usr/bin/env python3
"""
backend/src/segment_expansion_model.py

Cybba Segment Expansion MVP (offline, scalable)

Core idea:
- Use TF-IDF + cosine similarity to find underived Cybba components (Input B)
  that best "explain" competitor market segments (Input A).
- Produce proposed Cybba segments with clean naming:
    Cybba > L1 > L2 > Leaf

This version is "model-first" for taxonomy:
- L1 primarily comes from the learned L1 model (with confidence gating).
- Optional NN (nearest Cybba segment) can provide L1/L2 when similarity is strong.
- Path inference is OPTIONAL and only used if enabled and model/NN didn't produce L1.
- L2 is "open-world friendly": we prefer retrieval (`query/search`) over `predict()`
  so we don't lock into a closed set of labels.
- L2 catalog defaults / config defaults are OFF by default (can be enabled via cfg),
  because they reduce novelty and override what the model learned.

Key guardrails retained:
- ✅ Hard-block ABM/Top-Companies competitor segments
- ✅ Prevent duplicated suffixes like "Decision Makers Decision Makers"
- ✅ Strip level-code prefixes like "C: ", "D: ", "J: " from leaf
- ✅ Remove deterministic + hash/id tails from leaf
- ✅ Better B2B leaf construction (source-aware)
- ✅ Validator-safe sanitation (Cybba > L1 > L2 > Leaf)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from src.taxonomy_bundle import TaxonomyL1Bundle

# Make it discoverable under uvicorn multiprocessing module name
sys.modules.get("__mp_main__", sys).TaxonomyL1Bundle = TaxonomyL1Bundle
sys.modules.get("__main__", sys).TaxonomyL1Bundle = TaxonomyL1Bundle

try:
    import joblib
except ImportError:
    joblib = None


# ----------------------------
# Output schemas (ALWAYS defined)
# ----------------------------

PROPOSALS_COLS = [
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

COVERAGE_COLS = [
    "Competitor Provider",
    "Competitor Segment Name",
    "Competitor Segment ID",
    "Closest Cybba Segment",
    "Similarity",
]


# ----------------------------
# Helpers
# ----------------------------

_ACRONYMS = {"B2B", "ABM", "TV", "CPM", "USA", "US", "UK", "AFOL"}

_AUDIENCE_WORDS = {
    "shoppers", "buyers", "enthusiasts", "fans", "owners", "users", "parents",
    "travelers", "travellers", "households", "people", "individuals", "professionals",
    "students", "gamers", "runners", "viewers", "subscribers", "customers",
    "upgraders", "pilots", "note-takers", "notetakers", "collectors", "riders",
    "decision makers", "decision-maker",
}

_SUFFIXES = ["Decision Makers", "Buyers", "Enthusiasts", "Shoppers", "Fans", "Users"]

log = logging.getLogger("generator")
tax_log = logging.getLogger("taxonomy")


def _cfg_bool(cfg: dict, path: List[str], default: bool = False) -> bool:
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return bool(cur)


def clean(s: str) -> str:
    return str(s or "").strip()


def normalize_name(s: str) -> str:
    s = clean(s).lower()
    s = re.sub(r"[^\w\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def title_case_node(s: str) -> str:
    t = clean(s)
    if not t:
        return t
    if t.upper() in _ACRONYMS:
        return t.upper()
    return t.title()


def stable_suffix(component_ids: List[str], n: int = 8) -> str:
    key = "|".join(sorted(map(str, component_ids)))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:n]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def setup_logging(out_dir: Path, log_filename: str = "run.log") -> None:
    """
    Minimal, readable logging.
    - Console + file
    - Avoid duplicate handlers (uvicorn --reload)
    - LOG_LEVEL env var supported (default INFO)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    level_name = (os.environ.get("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Clear handlers to avoid duplicates on reload
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    fh = logging.FileHandler(out_dir / log_filename, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)


def _log_generator_stats(stats: Dict[str, int]) -> None:
    keys = [
        "total", "kept", "covered",
        "blocked_abm", "blocked_entity", "bad_leaf", "blocked_category",
        "skipped_missing_l1", "skipped_missing_l2",
        "l2_from_leaf", "l2_from_underived", "l2_from_catalog", "l2_from_cfg_default",
    ]
    msg = " ".join(f"{k}={int(stats.get(k, 0))}" for k in keys)
    log.info("[GEN] %s", msg)


def split_parts(name: str, sep: str) -> List[str]:
    s = clean(name)
    if not s:
        return []
    if ">" in sep:
        parts = re.split(r"\s*>\s*", s)
    else:
        parts = s.split(sep)
    return [p.strip() for p in parts if p and p.strip()]


def remove_trailing_deterministic(s: str) -> str:
    t = clean(s)
    return re.sub(r"\s*-\s*deterministic\s*$", "", t, flags=re.IGNORECASE).strip()


def strip_hash_code_suffix(s: str) -> str:
    """
    Remove garbage endings like:
      "0 - 656fedbb"
      "656fedbb"
      "... - deadbeef"
    """
    t = clean(s)

    # remove " - deadbeef"
    t = re.sub(r"\s*-\s*[a-f0-9]{6,}\s*$", "", t, flags=re.IGNORECASE).strip()

    # remove "0 - deadbeef" entire pattern
    if re.fullmatch(r"\d+\s*-\s*[a-f0-9]{6,}", t.lower()):
        return ""

    # if it became pure hex, blank it
    if re.fullmatch(r"[a-f0-9]{6,}", t.lower()):
        return ""

    return t


def strip_level_code_prefix(s: str) -> str:
    """
    Strip things like 'C: ', 'D: ', 'J: '.
    """
    t = clean(s)
    return re.sub(r"^[A-Z]\s*:\s*", "", t).strip()


def dedupe_repeated_suffix(text: str) -> str:
    """
    "Decision Makers Decision Makers" -> "Decision Makers"
    """
    t = clean(text)
    for suf in _SUFFIXES:
        t = re.sub(rf"(\b{re.escape(suf)}\b)(\s+\1)+$", r"\1", t, flags=re.IGNORECASE)
    return t.strip()


def is_bad_leaf(s: str) -> bool:
    t = clean(s)
    if not t:
        return True
    if re.fullmatch(r"\d+", t):
        return True
    if t.lower() in {"unknown", "na", "n/a", "none", "other"}:
        return True
    if re.fullmatch(r"[a-f0-9]{6,}", t.lower()):
        return True
    return False


def looks_like_audience_phrase(s: str) -> bool:
    tl = clean(s).lower()
    return any(w in tl for w in _AUDIENCE_WORDS)


def looks_like_brand_or_entity(s: str) -> bool:
    """
    Block obvious org/entity leaves.
    """
    t = clean(s)
    if not t:
        return True
    if re.search(r"\d{2,}", t):
        return True

    org_markers = [
        "university", "college", "school", "realty", "bank", "banker",
        "inc", "llc", "corp", "company", "ltd", "plc", "gmbh",
        "mutual", "insurance",
    ]
    if any(m in t.lower() for m in org_markers):
        return True

    return False


def is_abm_or_top_companies_segment(parts_lower: List[str]) -> bool:
    joined = " > ".join(parts_lower)
    if "top companies" in joined:
        return True
    if "abm" in parts_lower:
        return True
    return False


def ensure_audience_phrase(leaf: str, l1: str) -> str:
    t = title_case_node(clean(leaf))
    if not t:
        return ""

    if looks_like_audience_phrase(t):
        return t

    l1l = clean(l1).lower()
    if "purchase" in l1l:
        suffix = "Buyers"
    elif "behavior" in l1l:
        suffix = "Enthusiasts"
    elif "b2b" in l1l:
        if re.search(r"(director|vp|manager|engineer|c-suite|senior|lead|officer)", t, re.IGNORECASE):
            suffix = "Decision Makers"
        else:
            suffix = "Professionals"
    else:
        suffix = "Enthusiasts"

    if re.search(rf"\b{re.escape(suffix)}\b", t, flags=re.IGNORECASE):
        return t

    return f"{t} {suffix}"


def infer_l2_from_leaf(leaf: str) -> str:
    t = title_case_node(clean(leaf))
    if not t:
        return ""
    for suf in _SUFFIXES:
        if re.search(rf"\b{re.escape(suf)}\b$", t, flags=re.IGNORECASE):
            base = re.sub(rf"\s+\b{re.escape(suf)}\b$", "", t, flags=re.IGNORECASE).strip()
            base = title_case_node(base)
            if base and base.lower() != "none":
                return base
    return ""


def role_level_from_codes(raw_leaf: str) -> str:
    t = clean(raw_leaf)
    m = re.match(r"^([A-Z])\s*:\s*(.+)$", t)
    if not m:
        return strip_level_code_prefix(t)

    code = m.group(1).upper()
    role = clean(m.group(2))

    code_map = {
        "A": "C-Suite/C-Level",
        "B": "Vice President",
        "C": "Director",
        "D": "Manager",
        "J": role,
    }

    base = code_map.get(code, role) or role
    base = strip_level_code_prefix(base)

    if re.search(r"(c-suite|c-level|vice president|vp)", base, flags=re.IGNORECASE):
        return base

    return f"{base}-Level"


def derive_leaf_from_competitor_parts(parts: List[str], l1: str, *, enable_b2b_suffixes: bool) -> str:
    if not parts:
        return ""

    parts_lower = [p.lower() for p in parts]
    leaf = parts[-1]

    leaf = remove_trailing_deterministic(leaf)
    leaf = strip_hash_code_suffix(leaf)
    leaf = leaf.strip(" -").strip()
    leaf = dedupe_repeated_suffix(leaf)

    if "job seniority" in parts_lower:
        leaf = role_level_from_codes(parts[-1])
        leaf = title_case_node(leaf)
        if enable_b2b_suffixes and not re.search(r"\bdecision makers?\b", leaf, flags=re.IGNORECASE):
            leaf = f"{leaf} Decision Makers"
        return dedupe_repeated_suffix(leaf)

    if "functional area" in parts_lower:
        leaf = strip_level_code_prefix(leaf)
        leaf = title_case_node(leaf)

        try:
            fa_idx = parts_lower.index("functional area")
            after = parts[fa_idx + 1 :]
        except ValueError:
            after = parts[-2:]

        if len(after) >= 2:
            parent = title_case_node(after[-2])
            child = title_case_node(after[-1])
            if parent and child and parent.lower() not in child.lower():
                leaf = f"{parent} {child}"
            else:
                leaf = child

        if enable_b2b_suffixes and "b2b" in clean(l1).lower() and not re.search(r"\bdecision makers?\b", leaf, flags=re.IGNORECASE):
            leaf = f"{leaf} Decision Makers"

        return dedupe_repeated_suffix(leaf)

    if "industry" in parts_lower:
        leaf = strip_level_code_prefix(leaf)
        leaf = title_case_node(leaf)
        if enable_b2b_suffixes and "b2b" in clean(l1).lower() and not re.search(r"\bdecision makers?\b", leaf, flags=re.IGNORECASE):
            if re.search(r"\bindustry\b", leaf, flags=re.IGNORECASE):
                leaf = f"{leaf} Decision Makers"
            else:
                leaf = f"{leaf} Industry Decision Makers"
        return dedupe_repeated_suffix(leaf)

    if "decision makers" in parts_lower:
        leaf = strip_level_code_prefix(leaf)
        leaf = title_case_node(leaf)
        if enable_b2b_suffixes and not re.search(r"\bdecision makers?\b", leaf, flags=re.IGNORECASE):
            leaf = f"{leaf} Decision Makers"
        return dedupe_repeated_suffix(leaf)

    leaf = strip_level_code_prefix(leaf)
    leaf = title_case_node(leaf)
    return dedupe_repeated_suffix(leaf)


def normalize_l2_against_l1(l2: str, l1: str) -> str:
    l2t = title_case_node(clean(l2))
    l1t = title_case_node(clean(l1))
    if not l2t or not l1t:
        return l2t

    l1_norm = normalize_name(l1t)
    l2_norm = normalize_name(l2t)

    if l1_norm == l2_norm:
        return ""

    if l1_norm in l2_norm:
        pattern = r"\b" + re.escape(l1t) + r"\b"
        l2t = re.sub(pattern, "", l2t, flags=re.IGNORECASE)
        l2t = re.sub(r"\s{2,}", " ", l2t).strip()

    return title_case_node(l2t)


def _load_default_l2_by_l1_from_cfg(cfg: dict) -> Dict[str, str]:
    d = (cfg.get("taxonomy", {}) or {}).get("default_l2_by_l1", {}) or {}
    out: Dict[str, str] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            kk = title_case_node(clean(k))
            vv = title_case_node(clean(v))
            if kk:
                out[kk] = vv
    return out


def _load_l1_synonyms_from_cfg(cfg: dict) -> Dict[str, str]:
    syn = (cfg.get("taxonomy", {}) or {}).get("l1_synonyms", {}) or {}
    out: Dict[str, str] = {}
    if isinstance(syn, dict):
        for k, v in syn.items():
            kk = clean(k).lower()
            vv = clean(v)
            if kk:
                out[kk] = vv
    return out


def extract_cybba_l1_l2_leaf(name: str, sep: str = ">") -> Tuple[str, str, str, str]:
    parts = [p.strip() for p in clean(name).split(sep)]
    if len(parts) >= 4:
        provider = parts[0]
        l1 = parts[1]
        l2 = parts[2]
        leaf = " > ".join(parts[3:]).strip()
        return provider, title_case_node(l1), title_case_node(l2), leaf
    return "", "", "", ""


def build_taxonomy_catalog_from_cybba_segments(
    cybba_provider: str,
    A_cybba: pd.DataFrame,
    *,
    sep: str,
) -> Tuple[Dict[str, str], Dict[str, Counter]]:
    l2_counts_by_l1: Dict[str, Counter] = defaultdict(Counter)

    if A_cybba is None or len(A_cybba) == 0:
        return {}, {}

    for _, r in A_cybba.iterrows():
        name = clean(r.get("Segment Name", ""))
        if not name:
            continue
        prov, l1, l2, _leaf = extract_cybba_l1_l2_leaf(name, sep=">")
        if not prov or prov.lower() != cybba_provider.lower():
            continue
        if l1 and l2:
            l2_counts_by_l1[l1][l2] += 1

    most_common: Dict[str, str] = {}
    for l1, ctr in l2_counts_by_l1.items():
        if ctr:
            most_common[l1] = ctr.most_common(1)[0][0]

    return most_common, l2_counts_by_l1


def canonicalize_l1_l2(l1: str, l2: str, *, l1_synonyms: Dict[str, str]) -> Tuple[str, str]:
    raw_l1 = title_case_node(clean(l1))
    raw_l2 = title_case_node(clean(l2))

    l1_norm = clean(raw_l1).lower()
    if l1_norm and l1_norm in l1_synonyms:
        mapped = clean(l1_synonyms[l1_norm])
        raw_l1 = title_case_node(mapped) if mapped else ""

    return raw_l1, raw_l2


def infer_l1_from_path(parts_lower: List[str], default_l1: str, *, enable: bool) -> str:
    if not enable:
        return default_l1

    joined = " > ".join(parts_lower)

    purchase_keys = ["purchase", "in-market", "buyers", "shopping", "intent"]
    if any(k in joined for k in purchase_keys):
        return "Purchase Audience"

    demo_keys = ["demographic", "age", "gender", "income", "marital"]
    if any(k in joined for k in demo_keys):
        return "Demographic Audience"

    wealth_keys = ["wealth", "net worth", "affluent"]
    if any(k in joined for k in wealth_keys):
        return "Wealth Audience"

    edu_keys = ["education", "degree"]
    if any(k in joined for k in edu_keys):
        return "Education Audience"

    event_keys = ["event", "attendee", "conference", "ticket"]
    if any(k in joined for k in event_keys):
        return "Event Audience"

    b2b_keys = [
        "job seniority", "functional area", "job title", "job role",
        "department", "decision makers", "decision maker",
        "company size", "employee count", "company revenue", "firmographic",
        "industry",
    ]
    if any(k in joined for k in b2b_keys):
        return "B2B Audience"

    return default_l1


# ----------------------------
# Underived ID + compose
# ----------------------------

def component_id_from_underived_B(row: pd.Series, cfg: dict) -> str:
    strat = cfg["underived_id"]["strategy"]
    if strat == "field_value":
        fcol = cfg["underived_id"]["field_id_col"]
        vcol = cfg["underived_id"]["value_id_col"]
        f = clean(row.get(fcol, ""))
        v = clean(row.get(vcol, ""))
        if f and v:
            return f"{f}:{v}"

        sid_col = cfg["underived_id"].get("fallback_segment_id_col", "LiveRamp Segment ID")
        sid = clean(row.get(sid_col, ""))
        if sid:
            return sid

        seg_name = clean(row.get("Segment Name", ""))
        return "NAMEHASH:" + hashlib.md5(seg_name.encode("utf-8")).hexdigest()[:10]

    raise ValueError(f"Unsupported underived_id.strategy: {strat}")


def greedy_compose(
    target_vec,
    X_B,
    candidate_idxs: np.ndarray,
    *,
    max_components: int,
    top_m: int,
) -> Tuple[List[int], float]:
    sims = linear_kernel(target_vec, X_B[candidate_idxs]).ravel()
    order = np.argsort(-sims)[:min(top_m, len(candidate_idxs))]
    pool = candidate_idxs[order]

    selected: List[int] = []
    cur = None
    best_sim = -1.0

    for _ in range(max_components):
        best_j = None
        best_new_sim = best_sim

        for j in pool:
            new = X_B[j] if cur is None else (cur + X_B[j])
            s = float(linear_kernel(target_vec, new).ravel()[0])
            if s > best_new_sim + 1e-6:
                best_new_sim = s
                best_j = int(j)

        if best_j is None:
            break

        selected.append(best_j)
        cur = X_B[best_j] if cur is None else (cur + X_B[best_j])
        best_sim = best_new_sim

        pool = pool[pool != best_j]
        if len(pool) == 0:
            break

    return selected, float(best_sim)


# ----------------------------
# Taxonomy model loader + predictor
# ----------------------------

def _resolve_model_path(raw: str, *, base_dir: Optional[Path] = None) -> Path:
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p
    if base_dir is None:
        base_dir = Path.cwd()
    return (base_dir / p).resolve()


def load_taxonomy_models(cfg: dict) -> Tuple[Optional[object], Optional[object]]:
    if joblib is None:
        tax_log.warning("[TAX] joblib not installed; taxonomy disabled.")
        return None, None

    tax_cfg = cfg.get("taxonomy_model", {}) or {}
    enabled = bool(tax_cfg.get("enable", False))
    if not enabled:
        tax_log.info("[TAX] disabled")
        return None, None

    l1_raw = tax_cfg.get("l1_model_path")
    l2_raw = tax_cfg.get("l2_model_path")

    if l1_raw and l2_raw:
        base_dir = Path(__file__).resolve().parents[2]
        l1_path = _resolve_model_path(str(l1_raw), base_dir=base_dir)
        l2_path = _resolve_model_path(str(l2_raw), base_dir=base_dir)
    else:
        base_dir = Path(__file__).resolve().parents[2]
        model_dir = Path(cfg.get("training", {}).get("model_dir", "Data/Models"))
        if not model_dir.is_absolute():
            model_dir = (base_dir / model_dir).resolve()
        l1_path = model_dir / "cybba_taxonomy_L1.joblib"
        l2_path = model_dir / "cybba_taxonomy_L2.joblib"

    l1_model = joblib.load(l1_path) if l1_path.exists() else None
    l2_model = joblib.load(l2_path) if l2_path.exists() else None

    vmc = (cfg.get("taxonomy_model", {}) or {}).get("vertical_min_confidence", None)
    if vmc is not None and l1_model is not None and hasattr(l1_model, "vertical_min_confidence"):
        l1_model.vertical_min_confidence = float(vmc)

    l1_ok = bool(l1_path and l1_path.exists() and l1_model is not None)
    l2_ok = bool(l2_path and l2_path.exists() and l2_model is not None)

    tax_log.info("[TAX] loaded l1=%s l2=%s", l1_ok, l2_ok)
    if not l1_ok or not l2_ok:
        tax_log.warning("[TAX] missing model(s): L1=%s L2=%s", l1_path, l2_path)

    return l1_model, l2_model


def _extract_label_from_hits(hits) -> str:
    if hits is None:
        return ""
    if isinstance(hits, list) and hits:
        first = hits[0]
        if isinstance(first, list) and first:
            first = first[0]
        if isinstance(first, dict):
            return str(first.get("label") or first.get("L2") or first.get("name") or "")
        if isinstance(first, (tuple, list)) and len(first) >= 1:
            return str(first[0])
        if isinstance(first, str):
            return first
    if isinstance(hits, dict):
        return str(hits.get("label") or hits.get("L2") or hits.get("name") or "")
    return ""


def predict_l1_l2(l1_model, l2_model, text: str) -> Tuple[str, str]:
    """
    Model-first, open-world-friendly:
    - L1: l1_model.predict
    - L2: prefer retrieval/index (query/search) over predict(), so we're not locked
      into closed label sets.
    """
    if l1_model is None or l2_model is None:
        return "", ""

    X = [clean(text)]
    try:
        l1_raw = l1_model.predict(X)
        l1 = str(l1_raw[0]) if isinstance(l1_raw, (list, tuple, np.ndarray, pd.Series)) else str(l1_raw)

        hits = None
        if hasattr(l2_model, "query"):
            try:
                hits = l2_model.query(X, top_k=1)
            except Exception:
                hits = None
        elif hasattr(l2_model, "search"):
            try:
                hits = l2_model.search(X, top_k=1)
            except Exception:
                hits = None

        l2 = _extract_label_from_hits(hits)

        if not l2 and hasattr(l2_model, "predict"):
            try:
                l2_raw = l2_model.predict(X)
                l2 = str(l2_raw[0]) if isinstance(l2_raw, (list, tuple, np.ndarray, pd.Series)) else str(l2_raw)
            except Exception:
                l2 = ""

        return title_case_node(l1), title_case_node(l2)
    except Exception:
        return "", ""


def _apply_vertical_fallback_if_applicable(
    l1_model,
    X: List[str],
    l1_pred: List[str],
    l1_conf: List[Optional[float]],
) -> Tuple[List[str], List[Optional[float]]]:
    if not hasattr(l1_model, "vertical_model"):
        return l1_pred, l1_conf

    vmodel = getattr(l1_model, "vertical_model", None)
    if vmodel is None:
        return l1_pred, l1_conf

    b2b_label = str(getattr(l1_model, "b2b_label", "B2B Audience"))
    vmin = float(getattr(l1_model, "vertical_min_confidence", 0.55))

    idxs = [i for i, p in enumerate(l1_pred) if str(p) == b2b_label]
    if not idxs:
        return l1_pred, l1_conf

    Xb = [X[i] for i in idxs]
    if not hasattr(vmodel, "predict_proba"):
        return l1_pred, l1_conf

    vproba = vmodel.predict_proba(Xb)
    vclasses = getattr(vmodel, "classes_", None)
    if vclasses is None:
        return l1_pred, l1_conf

    best_idx = np.asarray(vproba).argmax(axis=1)
    best_conf = np.asarray(vproba).max(axis=1)
    best_label = [str(vclasses[j]) for j in best_idx]

    for local_i, global_i in enumerate(idxs):
        if float(best_conf[local_i]) >= vmin:
            l1_pred[global_i] = best_label[local_i]
            l1_conf[global_i] = float(best_conf[local_i])

    return l1_pred, l1_conf


# ----------------------------
# Public API: generate_proposals
# ----------------------------

def generate_proposals(
    cfg: dict,
    allowed_categories: Optional[List[str]] = None,
    *,
    progress_cb=None,
    progress_every: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = Path(cfg["paths"]["input_dir"])
    input_a_path = input_dir / cfg["files"]["input_a"]
    input_b_path = input_dir / cfg["files"]["input_b"]

    if not input_a_path.exists():
        raise FileNotFoundError(f"Input A not found: {input_a_path}")
    if not input_b_path.exists():
        raise FileNotFoundError(f"Input B not found: {input_b_path}")

    A = pd.read_csv(input_a_path, dtype=str, keep_default_na=False)
    B = pd.read_csv(input_b_path, dtype=str, keep_default_na=False)

    required = {"Provider Name", "Segment Name", "Segment Description"}
    if not required.issubset(A.columns):
        raise ValueError(f"Input A missing columns: {sorted(required - set(A.columns))}")
    if not required.issubset(B.columns):
        raise ValueError(f"Input B missing columns: {sorted(required - set(B.columns))}")

    cybba_provider = cfg["provider"]["cybba_name"]
    sep = cfg["taxonomy"]["separator"]

    l1_synonyms = _load_l1_synonyms_from_cfg(cfg)
    default_l2_by_l1_cfg = _load_default_l2_by_l1_from_cfg(cfg)

    enable_path_inference = _cfg_bool(cfg, ["taxonomy", "enable_path_inference"], default=False)
    enable_l2_catalog_fallback = _cfg_bool(cfg, ["taxonomy", "enable_l2_catalog_fallback"], default=False)
    enable_l2_cfg_default_fallback = _cfg_bool(cfg, ["taxonomy", "enable_l2_cfg_default_fallback"], default=False)

    gen_cfg = cfg.get("generation", {}) or {}
    use_gate = bool(gen_cfg.get("use_allowed_categories_gating", False))

    default_allowed = gen_cfg.get("allowed_categories_default", []) or []
    effective_allowed = allowed_categories if (allowed_categories and len(allowed_categories) > 0) else default_allowed
    effective_allowed = [str(x).strip() for x in (effective_allowed or []) if str(x).strip()]
    allowed_set = set(effective_allowed)

    fallback_l1 = gen_cfg.get("fallback_l1")
    fallback_l2 = gen_cfg.get("fallback_l2")

    max_proposals = int(cfg["generation"]["max_proposals"])
    max_components = int(cfg["generation"]["max_components"])
    retrieval_pool_size = int(cfg["generation"]["retrieval_pool_size"])

    log.info(
        "[GEN] start max_proposals=%d max_components=%d pool=%d gate=%s",
        max_proposals, max_components, retrieval_pool_size, use_gate
    )

    cover_enable = bool(cfg["coverage"].get("enable", True))
    cover_threshold = float(cfg["coverage"]["cover_threshold"])
    nn_min_sim = float(cfg.get("taxonomy", {}).get("nn_min_similarity", 0.35))

    A["text"] = (A["Segment Name"].astype(str) + " | " + A["Segment Description"].astype(str)).str.strip()
    B["text"] = (B["Segment Name"].astype(str) + " | " + B["Segment Description"].astype(str)).str.strip()

    tfidf_cfg = cfg["model"]["tfidf"]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=tfidf_cfg.get("stop_words", "english"),
        ngram_range=(int(tfidf_cfg.get("ngram_min", 1)), int(tfidf_cfg.get("ngram_max", 2))),
        min_df=int(tfidf_cfg.get("min_df", 2)),
        max_features=int(tfidf_cfg.get("max_features", 200000)),
    )
    X_B = vectorizer.fit_transform(B["text"].tolist())
    X_A = vectorizer.transform(A["text"].tolist())

    cybba_mask = A["Provider Name"].astype(str).str.lower() == cybba_provider.lower()
    A_cybba = A[cybba_mask].copy()
    A_other = A[~cybba_mask].copy()

    limit = int(cfg["generation"].get("max_candidates_from_A", 0))
    if limit and len(A_other) > limit:
        A_other = A_other.head(limit).copy()

    if cover_enable and len(A_cybba) > 0:
        X_A_cybba = X_A[A_cybba.index.values]
        cybba_norm_names = set(A_cybba["Segment Name"].astype(str).map(normalize_name))
    else:
        X_A_cybba = None
        cybba_norm_names = set()

    most_common_l2_by_l1, _ = build_taxonomy_catalog_from_cybba_segments(
        cybba_provider, A_cybba, sep=sep
    )

    l1_model, l2_model = load_taxonomy_models(cfg)

    proposals: List[Dict[str, str]] = []
    coverage_rows: List[Dict[str, str]] = []
    used_names = set(cybba_norm_names)

    stats: Dict[str, int] = {
        "total": 0,
        "covered": 0,
        "kept": 0,
        "bad_leaf": 0,
        "blocked_entity": 0,
        "blocked_abm": 0,
        "blocked_category": 0,
        "skipped_missing_l1": 0,
        "skipped_missing_l2": 0,
        "l2_from_leaf": 0,
        "l2_from_underived": 0,
        "l2_from_catalog": 0,
        "l2_from_cfg_default": 0,
        "leaf_has_decision_makers": 0,
    }

    total_candidates = int(len(A_other))

    def _emit_progress(force: bool = False):
        if not progress_cb:
            return
        cur = int(stats.get("total", 0))
        if force or (cur % max(int(progress_every), 1) == 0):
            progress_cb({
                "phase": "generate_proposals",
                "current": cur,
                "total": total_candidates,
                "kept": int(stats.get("kept", 0)),
                "covered": int(stats.get("covered", 0)),
                "blocked_abm": int(stats.get("blocked_abm", 0)),
                "blocked_entity": int(stats.get("blocked_entity", 0)),
                "blocked_category": int(stats.get("blocked_category", 0)),
            })

    for idx, a in A_other.iterrows():
        if len(proposals) >= max_proposals:
            break

        stats["total"] += 1
        _emit_progress()

        target_vec = X_A[idx]

        comp_parts = split_parts(a["Segment Name"], sep=sep)
        comp_parts_lower = [p.lower() for p in comp_parts]

        if is_abm_or_top_companies_segment(comp_parts_lower):
            stats["blocked_abm"] += 1
            continue

        # 1) Coverage shortcut
        best_match_name = ""
        best_match_sim = float("nan")

        if X_A_cybba is not None and len(A_cybba) > 0:
            sims = cosine_similarity(target_vec, X_A_cybba).ravel()
            best_i = int(np.argmax(sims))
            best_match_sim = float(sims[best_i])
            best_match_name = A_cybba.iloc[best_i]["Segment Name"]
            if best_match_sim >= cover_threshold:
                stats["covered"] += 1
                coverage_rows.append({
                    "Competitor Provider": a["Provider Name"],
                    "Competitor Segment Name": a["Segment Name"],
                    "Competitor Segment ID": a.get("LiveRamp Segment ID", ""),
                    "Closest Cybba Segment": best_match_name,
                    "Similarity": round(best_match_sim, 4) if not math.isnan(best_match_sim) else "",
                })
                continue

        # 2) Model prediction (taxonomy)
        early_text = f"{clean(a['Segment Name'])} | {clean(a['Segment Description'])}"
        X_early = [clean(early_text)]

        _l1_from_pair, l2_pred = predict_l1_l2(l1_model, l2_model, early_text)

        # L1 + confidence
        l1_pred_list: List[str] = [""]
        l1_conf_list: List[Optional[float]] = [None]

        try:
            if l1_model is not None and hasattr(l1_model, "predict"):
                raw = l1_model.predict(X_early)
                l1_pred_list = [str(raw[0])] if isinstance(raw, (list, tuple, np.ndarray, pd.Series)) else [str(raw)]

            if l1_model is not None and hasattr(l1_model, "predict_proba"):
                proba = l1_model.predict_proba(X_early)
                l1_conf_list = [float(np.asarray(proba).max())]
        except Exception:
            pass

        if l1_model is not None:
            l1_pred_list, l1_conf_list = _apply_vertical_fallback_if_applicable(
                l1_model, X_early, l1_pred_list, l1_conf_list
            )

        l1_pred = title_case_node(l1_pred_list[0]) if l1_pred_list else ""
        l1_conf = l1_conf_list[0] if l1_conf_list else None

        min_l1_conf = float(
            cfg.get("taxonomy", {}).get(
                "l1_min_confidence",
                cfg.get("taxonomy_model", {}).get("l1_min_confidence", 0.70),
            )
        )
        if l1_pred and (l1_conf is not None) and (l1_conf < min_l1_conf):
            l1_pred = ""

        # 3) Optional NN taxonomy
        nn_l1, nn_l2 = "", ""
        if best_match_name and (not math.isnan(best_match_sim)) and best_match_sim >= nn_min_sim:
            _, nn_l1, nn_l2, _ = extract_cybba_l1_l2_leaf(best_match_name, sep=">")

        # 4) Optional path inference
        l1_path = ""
        if enable_path_inference and not l1_pred and not nn_l1:
            l1_path = infer_l1_from_path(comp_parts_lower, default_l1="", enable=True)

        l1 = l1_pred or nn_l1 or l1_path
        l2 = nn_l2 or l2_pred

        l1, l2 = canonicalize_l1_l2(l1, l2, l1_synonyms=l1_synonyms)
        l2 = normalize_l2_against_l1(l2, l1)

        if use_gate and allowed_set and l1 and (l1 not in allowed_set):
            stats["blocked_category"] += 1
            continue

        if not l1:
            if fallback_l1:
                l1 = title_case_node(fallback_l1)
            else:
                stats["skipped_missing_l1"] += 1
                continue

        # Prefilter candidates in B
        sims_to_all = linear_kernel(target_vec, X_B).ravel()
        topK = int(cfg["generation"].get("prefilter_top_k", 2000))
        topK = min(max(topK, 1), len(sims_to_all))
        cand = np.argpartition(-sims_to_all, kth=topK - 1)[:topK]
        candidate_idxs = cand.astype(int)

        chosen_idxs, score = greedy_compose(
            target_vec,
            X_B,
            candidate_idxs,
            max_components=max_components,
            top_m=retrieval_pool_size,
        )
        if not chosen_idxs:
            continue

        min_sim = float(cfg.get("generation", {}).get("min_composition_similarity", 0.0) or 0.0)
        if float(score) < min_sim:
            continue

        comps = B.iloc[chosen_idxs].copy()
        comp_ids = [component_id_from_underived_B(r, cfg) for _, r in comps.iterrows()]
        comp_names = comps["Segment Name"].tolist()

        enable_suffixes = bool((cfg.get("taxonomy", {}) or {}).get("enable_b2b_leaf_suffixes", False))
        leaf_raw = derive_leaf_from_competitor_parts(comp_parts, l1=l1, enable_b2b_suffixes=enable_suffixes)

        if looks_like_brand_or_entity(leaf_raw):
            stats["blocked_entity"] += 1
            bparts = split_parts(comps.iloc[0]["Segment Name"], sep=sep)
            leaf_raw2 = derive_leaf_from_competitor_parts(bparts, l1=l1, enable_b2b_suffixes=enable_suffixes)
            if looks_like_brand_or_entity(leaf_raw2):
                stats["bad_leaf"] += 1
                continue
            leaf_raw = leaf_raw2

        leaf_raw = dedupe_repeated_suffix(leaf_raw)
        if is_bad_leaf(leaf_raw):
            stats["bad_leaf"] += 1
            continue

        leaf_final = ensure_audience_phrase(leaf_raw, l1)
        if not leaf_final:
            stats["bad_leaf"] += 1
            continue

        if "decision maker" in leaf_final.lower():
            stats["leaf_has_decision_makers"] += 1

        # Final validator-safe cleanup
        l1 = title_case_node(clean(l1).replace(">", " "))
        l2 = title_case_node(clean(l2).replace(">", " "))
        leaf_final = title_case_node(clean(leaf_final).replace(">", " "))

        # -------------------------
        # L2 Derivation (novelty-preserving)
        # -------------------------
        if not l2:
            l2_from_leaf = infer_l2_from_leaf(leaf_final)
            l2_from_leaf = normalize_l2_against_l1(l2_from_leaf, l1)
            if l2_from_leaf and (l2_from_leaf.lower() != l1.lower()):
                l2 = l2_from_leaf
                stats["l2_from_leaf"] += 1

        if not l2 and len(comps) > 0:
            bparts = split_parts(comps.iloc[0]["Segment Name"], sep=sep)
            b_leaf = derive_leaf_from_competitor_parts(bparts, l1=l1, enable_b2b_suffixes=enable_suffixes)
            b_leaf = ensure_audience_phrase(b_leaf, l1)
            b_leaf = title_case_node(clean(b_leaf))
            l2_underived = infer_l2_from_leaf(b_leaf)
            if l2_underived and (l2_underived.lower() != l1.lower()):
                l2 = l2_underived
                stats["l2_from_underived"] += 1

        if not l2 and enable_l2_catalog_fallback:
            l2_cat = most_common_l2_by_l1.get(l1, "")
            if l2_cat and (l2_cat.lower() != l1.lower()):
                l2 = l2_cat
                stats["l2_from_catalog"] += 1

        if not l2 and enable_l2_cfg_default_fallback:
            l2_cfg = default_l2_by_l1_cfg.get(l1, "")
            if l2_cfg and (l2_cfg.lower() != l1.lower()):
                l2 = l2_cfg
                stats["l2_from_cfg_default"] += 1

        if not l2:
            if fallback_l2:
                l2 = title_case_node(fallback_l2)
            else:
                stats["skipped_missing_l2"] += 1
                continue

        proposed_name = f"{cybba_provider} > {l1} > {l2} > {leaf_final}"

        nn_name = normalize_name(proposed_name)
        if nn_name in used_names and cfg["dedupe"].get("resolve_name_collision", True):
            proposed_name = f"{proposed_name} - {stable_suffix(comp_ids, n=int(cfg['dedupe'].get('collision_hash_len', 8)))}"
            nn_name = normalize_name(proposed_name)

        used_names.add(nn_name)

        proposals.append({
            "Competitor Provider": a["Provider Name"],
            "Competitor Segment Name": a["Segment Name"],
            "Competitor Segment ID": a.get("LiveRamp Segment ID", ""),
            "Proposed New Segment Name": proposed_name,
            "Non Derived Segments utilized": "; ".join([f"{cid} | {nm}" for cid, nm in zip(comp_ids, comp_names)]),
            "Composition Similarity": round(score, 4),
            "Closest Cybba Segment": best_match_name,
            "Closest Cybba Similarity": round(best_match_sim, 4) if not math.isnan(best_match_sim) else "",
            "Taxonomy": f"{l1} > {l2}",
        })
        stats["kept"] += 1

    _log_generator_stats(stats)
    _emit_progress(force=True)

    proposals_df = pd.DataFrame(proposals, columns=PROPOSALS_COLS)
    coverage_df = pd.DataFrame(coverage_rows, columns=COVERAGE_COLS)
    return proposals_df, coverage_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "config.yml"),
        help="Path to YAML config file",
    )
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    output_dir = Path(cfg["paths"]["output_dir"])
    setup_logging(out_dir=output_dir, log_filename=cfg.get("output", {}).get("log_filename", "run.log"))

    proposals_df, coverage_df = generate_proposals(cfg, allowed_categories=None)

    proposals_path = output_dir / cfg["output"]["proposals_filename"]
    proposals_df.to_csv(proposals_path, index=False, encoding="utf-8")
    log.info("[GEN] wrote proposals rows=%d file=%s", len(proposals_df), proposals_path)

    if len(coverage_df) > 0:
        coverage_path = output_dir / cfg["output"]["coverage_filename"]
        coverage_df.to_csv(coverage_path, index=False, encoding="utf-8")
        log.info("[GEN] wrote coverage rows=%d file=%s", len(coverage_df), coverage_path)


if __name__ == "__main__":
    main()