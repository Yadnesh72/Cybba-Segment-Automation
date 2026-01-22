#!/usr/bin/env python3
"""
backend/src/segment_expansion_model.py

Cybba Segment Expansion MVP (offline, scalable)

Core idea:
- Use TF-IDF + cosine similarity to find underived Cybba components (Input B)
  that best "explain" competitor market segments (Input A).
- Produce proposed Cybba segments with clean naming:
    Cybba > L1 > L2 > Leaf

Key upgrade:
- Uses trained taxonomy models (joblib) to predict L1/L2 in the Output_format.csv style.

Fixes in this update (IMPORTANT):
- ✅ Fix greedy_compose bug (cur vector used wrong index)
- ✅ Hard-block ABM/Top-Companies competitor segments (no brand/entity audiences)
- ✅ Prevent duplicated suffixes like "Decision Makers Decision Makers"
- ✅ Strip level-code prefixes like "C: ", "D: ", "J: " from leaf
- ✅ Remove deterministic + hash/id tails from leaf
- ✅ Better B2B leaf construction (source-aware):
    - Job Seniority -> "Director-Level Decision Makers", etc. (reduces collisions)
    - Functional Area -> can include parent context to reduce collisions
    - Industry -> keeps "Industry" phrasing
- ✅ Taxonomy guardrail:
    If leaf contains "Decision Makers" -> force L1="B2B Audience"
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(out_dir / log_filename, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


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
    Strip things like 'C: ', 'D: ', 'J: ' (common in firmographic role/revenue codes).
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
    Block obvious org/entity leaves (prevents 'Coldwell Banker Enthusiasts', 'Ohio State University Enthusiasts').
    Conservative markers — tune as needed.
    """
    t = clean(s)
    if not t:
        return True

    # heavy digits => often codes/brands
    if re.search(r"\d{2,}", t):
        return True

    org_markers = [
        "university", "college", "school", "realty", "bank", "banker",
        "inc", "llc", "corp", "company", "ltd", "plc", "gmbh",
        "mutual", "insurance",  # common ABM/company strings
    ]
    if any(m in t.lower() for m in org_markers):
        return True

    return False


def is_abm_or_top_companies_segment(parts_lower: List[str]) -> bool:
    """
    Hard-block ABM/Top-Companies competitor segments. These are entity lists, not audiences.
    """
    joined = " > ".join(parts_lower)
    if "top companies" in joined:
        return True
    if "abm" in parts_lower:
        return True
    return False


def ensure_audience_phrase(leaf: str, l1: str) -> str:
    """
    Make leaf match Output_format.csv style:
    - if already has audience words -> keep
    - else add a suffix based on L1 category
    - ✅ don't double-append
    """
    t = title_case_node(clean(leaf))
    if not t:
        return ""

    # already audience-like → keep
    if looks_like_audience_phrase(t):
        return t

    l1l = clean(l1).lower()
    if "purchase" in l1l:
        suffix = "Buyers"
    elif "behavior" in l1l:
        suffix = "Enthusiasts"
    elif "b2b" in l1l:
        suffix = "Decision Makers"
    else:
        suffix = "Enthusiasts"

    # don't double-append
    if re.search(rf"\b{re.escape(suffix)}\b", t, flags=re.IGNORECASE):
        return t

    return f"{t} {suffix}"


def force_b2b_if_decision_makers(l1: str, leaf: str) -> str:
    """
    Guardrail: if the leaf contains Decision Makers, this must be a B2B audience.
    """
    if re.search(r"\bdecision makers?\b", clean(leaf), flags=re.IGNORECASE):
        return "B2B Audience"
    return l1


def role_level_from_codes(raw_leaf: str) -> str:
    """
    Turn "C: Director" / "D: Manager" style leaves into "Director-Level" / "Manager-Level".
    If the raw leaf doesn't look coded, return cleaned value.
    """
    t = clean(raw_leaf)

    # Preserve the code if present at the start
    m = re.match(r"^([A-Z])\s*:\s*(.+)$", t)
    if not m:
        return strip_level_code_prefix(t)

    code = m.group(1).upper()
    role = clean(m.group(2))

    # lightweight mapping (you can tune wording later)
    code_map = {
        "A": "C-Suite/C-Level",
        "B": "Vice President",
        "C": "Director",
        "D": "Manager",
        "J": role,  # revenue codes etc; just fall back to role text
    }

    base = code_map.get(code, role) or role
    base = strip_level_code_prefix(base)

    # If it's already something like "C-Suite/C-Level", don't add "-Level"
    if re.search(r"(c-suite|c-level|vice president|vp)", base, flags=re.IGNORECASE):
        return base

    return f"{base}-Level"


def derive_leaf_from_competitor_parts(parts: List[str], l1: str) -> str:
    """
    Source-aware leaf creation to reduce collisions and remove junk.

    Examples:
      Job Seniority > C: Director                -> Director-Level Decision Makers
      Functional Area > Marketing > Advertising  -> Marketing Advertising Decision Makers (adds context)
      Industry > Real Estate Industry            -> Real Estate Industry Decision Makers
      Decision Makers > Finance Decision Makers  -> Finance Decision Makers  (keeps as-is)
    """
    if not parts:
        return ""

    parts_lower = [p.lower() for p in parts]

    # base leaf: last node
    leaf = parts[-1]

    # Normalize common junk on leaf
    leaf = remove_trailing_deterministic(leaf)
    leaf = strip_hash_code_suffix(leaf)
    leaf = leaf.strip(" -").strip()
    leaf = dedupe_repeated_suffix(leaf)

    # Job Seniority special handling
    if "job seniority" in parts_lower:
        # leaf may be "C: Director"
        leaf = role_level_from_codes(parts[-1])  # keeps coded meaning
        leaf = title_case_node(leaf)
        # ensure it's B2B decision makers style
        if not re.search(r"\bdecision makers?\b", leaf, flags=re.IGNORECASE):
            leaf = f"{leaf} Decision Makers"
        return dedupe_repeated_suffix(leaf)

    # Functional Area: optionally add parent context to reduce collisions
    if "functional area" in parts_lower:
        leaf = strip_level_code_prefix(leaf)
        leaf = title_case_node(leaf)

        # If there is a parent after "Functional Area", include it for context when helpful
        try:
            fa_idx = parts_lower.index("functional area")
            after = parts[fa_idx + 1 :]
        except ValueError:
            after = parts[-2:]

        # after could be ["Marketing", "Advertising"] -> include both
        if len(after) >= 2:
            parent = title_case_node(after[-2])
            child = title_case_node(after[-1])
            # avoid doubling
            if parent and child and parent.lower() not in child.lower():
                leaf = f"{parent} {child}"
            else:
                leaf = child

        # B2B: default to decision makers unless leaf already has an audience suffix
        if "b2b" in clean(l1).lower() and not re.search(r"\bdecision makers?\b", leaf, flags=re.IGNORECASE):
            leaf = f"{leaf} Decision Makers"

        return dedupe_repeated_suffix(leaf)

    # Industry: keep "Industry" phrasing
    if "industry" in parts_lower:
        leaf = strip_level_code_prefix(leaf)
        leaf = title_case_node(leaf)
        if "b2b" in clean(l1).lower() and not re.search(r"\bdecision makers?\b", leaf, flags=re.IGNORECASE):
            # If it already ends with Industry, keep it
            if re.search(r"\bindustry\b", leaf, flags=re.IGNORECASE):
                leaf = f"{leaf} Decision Makers"
            else:
                leaf = f"{leaf} Industry Decision Makers"
        return dedupe_repeated_suffix(leaf)

    # Decision Makers: often already includes suffix; just normalize
    if "decision makers" in parts_lower:
        leaf = strip_level_code_prefix(leaf)
        leaf = title_case_node(leaf)
        if not re.search(r"\bdecision makers?\b", leaf, flags=re.IGNORECASE):
            leaf = f"{leaf} Decision Makers"
        return dedupe_repeated_suffix(leaf)

    # Company Location Type: keep leaf as-is and add Decision Makers if B2B
    if "company location type" in parts_lower:
        leaf = strip_level_code_prefix(leaf)
        leaf = title_case_node(leaf)
        if "b2b" in clean(l1).lower() and not re.search(r"\bdecision makers?\b", leaf, flags=re.IGNORECASE):
            leaf = f"{leaf} Decision Makers"
        return dedupe_repeated_suffix(leaf)

    # Default: just cleaned leaf
    leaf = strip_level_code_prefix(leaf)
    leaf = title_case_node(leaf)
    return dedupe_repeated_suffix(leaf)


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

        return f"NAMEHASH:{hashlib.md5(clean(row.get('Segment Name','')).encode('utf-8')).hexdigest()[:10]}"

    raise ValueError(f"Unsupported underived_id.strategy: {strat}")


def greedy_compose(
    target_vec,
    X_B,
    candidate_idxs: np.ndarray,
    *,
    max_components: int,
    top_m: int,
) -> Tuple[List[int], float]:
    """
    Greedy add up to max_components underived segments (from B) to best match target_vec.
    ✅ FIXED: uses best_j when updating cur (not stale loop var j)
    """
    sims = cosine_similarity(target_vec, X_B[candidate_idxs]).ravel()
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
            s = float(cosine_similarity(target_vec, new).ravel()[0])
            if s > best_new_sim + 1e-6:
                best_new_sim = s
                best_j = int(j)

        if best_j is None:
            break

        selected.append(best_j)
        cur = X_B[best_j] if cur is None else (cur + X_B[best_j])
        best_sim = best_new_sim

    return selected, float(best_sim)


# ----------------------------
# Taxonomy model loader + predictor
# ----------------------------

def load_taxonomy_models(cfg: dict) -> Tuple[Optional[object], Optional[object]]:
    """
    Loads joblib models trained from Output_format.csv.
    Returns (l1_model, l2_model). Either may be None.
    """
    if joblib is None:
        logging.warning("joblib not installed; taxonomy models will not load.")
        return None, None

    model_dir = Path(cfg.get("training", {}).get("model_dir", "Data/Models"))
    l1_path = model_dir / "cybba_taxonomy_L1.joblib"
    l2_path = model_dir / "cybba_taxonomy_L2.joblib"

    l1_model = joblib.load(l1_path) if l1_path.exists() else None
    l2_model = joblib.load(l2_path) if l2_path.exists() else None

    if l1_model is None or l2_model is None:
        logging.warning(
            "Taxonomy models not found at %s (L1=%s, L2=%s).",
            model_dir, l1_path.exists(), l2_path.exists()
        )
    else:
        logging.info("Loaded taxonomy models from %s", model_dir)

    return l1_model, l2_model


def predict_l1_l2(l1_model, l2_model, text: str) -> Tuple[str, str]:
    """
    Predict (L1, L2) using trained models.
    """
    if l1_model is None or l2_model is None:
        return "", ""
    X = [clean(text)]
    try:
        l1 = str(l1_model.predict(X)[0])
        l2 = str(l2_model.predict(X)[0])
        return title_case_node(l1), title_case_node(l2)
    except Exception:
        return "", ""


# ----------------------------
# Public API: generate_proposals
# ----------------------------

def generate_proposals(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    max_proposals = int(cfg["generation"]["max_proposals"])
    max_components = int(cfg["generation"]["max_components"])
    retrieval_pool_size = int(cfg["generation"]["retrieval_pool_size"])

    cover_enable = bool(cfg["coverage"].get("enable", True))
    cover_threshold = float(cfg["coverage"]["cover_threshold"])

    # features
    A["text"] = (A["Segment Name"].astype(str) + " | " + A["Segment Description"].astype(str)).str.strip()
    B["text"] = (B["Segment Name"].astype(str) + " | " + B["Segment Description"].astype(str)).str.strip()

    # TF-IDF (fit on B, transform A)
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

    # split A into Cybba vs others
    cybba_mask = A["Provider Name"].astype(str).str.lower() == cybba_provider.lower()
    A_cybba = A[cybba_mask].copy()
    A_other = A[~cybba_mask].copy()

    # optional limiter (dev only)
    limit = int(cfg["generation"].get("max_candidates_from_A", 0))
    if limit and len(A_other) > limit:
        A_other = A_other.head(limit).copy()

    # coverage store
    if cover_enable and len(A_cybba) > 0:
        X_A_cybba = X_A[A_cybba.index.values]
        cybba_norm_names = set(A_cybba["Segment Name"].astype(str).map(normalize_name))
    else:
        X_A_cybba = None
        cybba_norm_names = set()

    # taxonomy models
    l1_model, l2_model = load_taxonomy_models(cfg)

    proposals: List[Dict[str, str]] = []
    coverage_rows: List[Dict[str, str]] = []
    used_names = set(cybba_norm_names)

    candidate_idxs_all = np.arange(len(B), dtype=int)

    stats = {
        "total": 0,
        "covered": 0,
        "kept": 0,
        "bad_leaf": 0,
        "no_tax_pred": 0,
        "blocked_entity": 0,
        "blocked_abm": 0,
    }

    for idx, a in A_other.iterrows():
        if len(proposals) >= max_proposals:
            break

        stats["total"] += 1
        target_vec = X_A[idx]

        # competitor taxonomy parts (used for ABM blocking + better leaf)
        comp_parts = split_parts(a["Segment Name"], sep=sep)
        comp_parts_lower = [p.lower() for p in comp_parts]

        # ✅ Hard block ABM/top-companies
        if is_abm_or_top_companies_segment(comp_parts_lower):
            stats["blocked_abm"] += 1
            continue

        # 1) coverage check
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

        # 2) compose from underived B
        chosen_idxs, score = greedy_compose(
            target_vec,
            X_B,
            candidate_idxs_all,
            max_components=max_components,
            top_m=retrieval_pool_size,
        )
        if not chosen_idxs:
            continue

        comps = B.iloc[chosen_idxs].copy()
        comp_ids = [component_id_from_underived_B(r, cfg) for _, r in comps.iterrows()]
        comp_names = comps["Segment Name"].tolist()

        # 3) predict L1/L2 using taxonomy model (learned from Output_format.csv)
        pred_text = clean(a["Segment Name"]) + " | " + clean(a["Segment Description"])
        l1, l2 = predict_l1_l2(l1_model, l2_model, pred_text)
        if not l1 or not l2:
            stats["no_tax_pred"] += 1
            l1, l2 = "Behavioral Audience", "Lifestyle"

        # 4) leaf (source-aware, collision-reducing, clean)
        leaf_raw = derive_leaf_from_competitor_parts(comp_parts, l1=l1)

        # If competitor leaf looks like an entity, try underived leaf fallback
        if looks_like_brand_or_entity(leaf_raw):
            stats["blocked_entity"] += 1
            bparts = split_parts(comps.iloc[0]["Segment Name"], sep=sep)
            leaf_raw2 = derive_leaf_from_competitor_parts(bparts, l1=l1)

            if looks_like_brand_or_entity(leaf_raw2):
                stats["bad_leaf"] += 1
                continue
            leaf_raw = leaf_raw2

        leaf_raw = dedupe_repeated_suffix(leaf_raw)

        if is_bad_leaf(leaf_raw):
            stats["bad_leaf"] += 1
            continue

        # Audience suffix normalization (only appends if missing)
        leaf_final = ensure_audience_phrase(leaf_raw, l1)
        if not leaf_final:
            stats["bad_leaf"] += 1
            continue

        # ✅ Guardrail: decision makers => force B2B Audience
        l1 = force_b2b_if_decision_makers(l1, leaf_final)

        proposed_name = f"{cybba_provider} > {l1} > {l2} > {leaf_final}"

        # collision dedupe (keep as last resort)
        nn = normalize_name(proposed_name)
        if nn in used_names and cfg["dedupe"].get("resolve_name_collision", True):
            proposed_name = f"{proposed_name} - {stable_suffix(comp_ids, n=int(cfg['dedupe'].get('collision_hash_len', 8)))}"
            nn = normalize_name(proposed_name)
        used_names.add(nn)

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

    logging.info("GENERATOR STATS: %s", stats)

    proposals_df = pd.DataFrame(proposals, columns=PROPOSALS_COLS)
    coverage_df = pd.DataFrame(coverage_rows, columns=COVERAGE_COLS)
    return proposals_df, coverage_df


# ----------------------------
# Script entrypoint (optional)
# ----------------------------

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
    setup_logging(output_dir, log_filename=cfg.get("output", {}).get("log_filename", "run.log"))

    proposals_df, coverage_df = generate_proposals(cfg)

    proposals_path = output_dir / cfg["output"]["proposals_filename"]
    proposals_df.to_csv(proposals_path, index=False, encoding="utf-8")
    logging.info("Wrote proposals: %s (rows=%d)", proposals_path, len(proposals_df))

    if len(coverage_df) > 0:
        coverage_path = output_dir / cfg["output"]["coverage_filename"]
        coverage_df.to_csv(coverage_path, index=False, encoding="utf-8")
        logging.info("Wrote coverage: %s (rows=%d)", coverage_path, len(coverage_df))


if __name__ == "__main__":
    main()
