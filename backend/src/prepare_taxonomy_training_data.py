#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List, Dict, Any


import pandas as pd

# ------------------------------------------------------------
# Defaults so the script can be run with no args
# ------------------------------------------------------------
DEFAULT_CATALOG_PATH = "/Users/yadnesh/cybba/cybba_segment_automation/Data/Input/Raw_segments/Data_Marketplace_Full_Catalog.csv"
DEFAULT_OUTPUT_PATH = "/Users/yadnesh/cybba/cybba_segment_automation/Data/Input/Raw_segments/taxonomy_training_prepared.csv.gz"


def clean(s: str) -> str:
    return str(s or "").strip()


def strip_leading_asterisks(s: str) -> str:
    # fixes "**AlarisB2B" -> "AlarisB2B"
    return re.sub(r"^\*+\s*", "", clean(s))


def collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", clean(s)).strip()


def normalize_node(s: str) -> str:
    # trim + collapse spaces + strip leading asterisks
    t = strip_leading_asterisks(s)
    t = collapse_spaces(t)

    # normalize weird separators
    t = t.replace("–", "-").replace("—", "-")

    # remove obvious trailing junk separators
    t = re.sub(r"\s*[\|\-–—]+\s*$", "", t).strip()
    return t


def split_parts(name: str) -> List[str]:
    """
    Robust split on '>' regardless of whitespace.
    Also strips leading asterisks on each node.
    """
    s = clean(name)
    if not s:
        return []
    parts = [p.strip() for p in re.split(r"\s*>\s*", s) if p and p.strip()]
    parts = [normalize_node(p) for p in parts]
    return [p for p in parts if p]


def _lower(s: str) -> str:
    return normalize_node(s).lower()


def _tokenize(s: str) -> List[str]:
    # simple tokens for heuristics
    return [t for t in re.split(r"[^a-z0-9]+", _lower(s)) if t]


def looks_like_category(node: str) -> bool:
    """
    Category-ish nodes tend to be:
    - short/medium length
    - contain generic words (audience, intent, purchase, etc.)
    """
    n = _lower(node)
    if not n:
        return False

    generic = {
    "audience", "audiences", "intent", "purchase", "purchases", "demographic",
    "behavior", "behaviour", "behavioral", "transaction", "transactional",
    "automotive", "health", "finance", "insurance", "technology", "education",
    "wealth", "events", "lifestyle", "interests", "curated", "custom",
    "industry", "company", "household", "consumer", "buyers",
    "shoppers", "owners", "students", "parents", "decision", "makers",
    "b2b", "b2c"
}

    toks = set(_tokenize(n))
    if toks & generic:
        return True

    # too long => more likely a leaf than a category
    if len(n) > 45:
        return False

    # many words => more likely a leaf than a category
    if len(n.split()) >= 6:
        return False

    return True


def looks_like_root_bucket(node: str, provider: str, learned_roots: set[str]) -> bool:
    """
    Dynamic root bucket detection.
    """
    n = _lower(node)
    p = _lower(provider)

    if not n:
        return False

    # provider-ish
    if p and (n == p or p in n or n in p):
        return True

    # learned roots
    if n in learned_roots:
        return True

    # wrapper markers
    wrapper_markers = {"segments", "segment", "taxonomy", "marketplace", "audience", "audiences"}
    if any(w in n for w in wrapper_markers):
        return True

    # vendor-ish suffix
    if n.endswith("b2b") or n.endswith("people") or n.endswith("segments"):
        return True

    # brandy root: one token, fairly long
    if " " not in n and len(n) >= 7:
        return True

    return False


def learn_provider_roots(df: pd.DataFrame) -> Dict[str, set[str]]:
    """
    Learns per-provider root bucket candidates by scanning segment names.
    """
    roots: Dict[str, Dict[str, int]] = {}

    for _, r in df.iterrows():
        provider = normalize_node(r.get("Provider Name", ""))
        seg = r.get("Segment Name", "")
        parts = split_parts(seg)
        if not provider or not parts:
            continue

        if parts and _lower(parts[0]) == _lower(provider):
            parts = parts[1:]
        if not parts:
            continue

        first = _lower(parts[0])
        if not first:
            continue

        roots.setdefault(provider, {})
        roots[provider][first] = roots[provider].get(first, 0) + 1

    learned: Dict[str, set[str]] = {}
    for provider, counts in roots.items():
        items = sorted(counts.items(), key=lambda x: -x[1])
        if not items:
            learned[provider] = set()
            continue

        total = sum(c for _, c in items)
        out = set()

        for node, c in items[:50]:
            frac = c / max(total, 1)
            if c >= 200 or frac >= 0.05:
                if not looks_like_category(node):
                    out.add(node)

        learned[provider] = out

    return learned


# Drop only explicit junk trees
_JUNK_TREE_PATTERNS = [
    r"\babm\b",
    r"\baccount[-\s]*based\b",
    r"\baccount\s*list(s)?\b",
    r"\bnamed\s*accounts\b",
    r"\btarget\s*accounts\b",
    r"\btop\s*accounts\b",
    r"\bcompany\s*list(s)?\b",
    r"\bfirm\s*list(s)?\b",
]
_TOP_COMPANY_PATTERN = r"\btop\s*compan(y|ies)\b"


def should_drop_tree(full_parts: List[str], drop_abm: bool, drop_top_companies: bool) -> bool:
    joined = " > ".join(_lower(p) for p in full_parts)

    if drop_top_companies and re.search(_TOP_COMPANY_PATTERN, joined):
        return True

    if drop_abm:
        for pat in _JUNK_TREE_PATTERNS:
            if re.search(pat, joined):
                return True

    return False


def _apply_root_shifts(parts: List[str], provider: str, learned_roots: set[str], max_shifts: int = 2) -> List[str]:
    out = parts
    for _shift in range(max_shifts):
        if len(out) >= 4 and looks_like_root_bucket(out[0], provider, learned_roots):
            out = out[1:]
        else:
            break
    return out


def _candidate_l1_after_root_shift(provider: str, parts: List[str], learned_roots_by_provider: Dict[str, set[str]]) -> str:
    learned_roots = learned_roots_by_provider.get(provider, set())
    shifted = _apply_root_shifts(parts, provider, learned_roots, max_shifts=2)
    if len(shifted) < 2:
        return ""
    return normalize_node(shifted[0])


def _learn_l1_frequencies_after_root_shift(
    df: pd.DataFrame,
    learned_roots_by_provider: Dict[str, set[str]],
    min_depth: int,
) -> Dict[str, Dict[str, int]]:
    freq: Dict[str, Dict[str, int]] = {}

    for _, r in df.iterrows():
        provider = normalize_node(r.get("Provider Name", ""))
        seg = r.get("Segment Name", "")
        parts = split_parts(seg)
        if not provider or not parts:
            continue

        if parts and _lower(parts[0]) == _lower(provider):
            parts = parts[1:]
        if len(parts) < min_depth:
            continue

        l1 = _candidate_l1_after_root_shift(provider, parts, learned_roots_by_provider)
        if not l1:
            continue

        freq.setdefault(provider, {})
        freq[provider][l1] = freq[provider].get(l1, 0) + 1

    return freq


# ------------------------------------------------------------
# ✅ Option B: Intelligent Taxonomy Growth
# ------------------------------------------------------------
_BAD_L1_PATTERNS = [
    r"\bpowered by\b",
    r"\bcookies?\b",
    r"\bids?\b",
    r"\bmaid\b",
    r"\bxandr\b",
    r"\btradedesk\b",
    r"\blotame\b",
    r"\bacxiom\b",
    r"\bexelate\b",
    r"\bnielsen\b",
    r"\bcircana\b",
    r"\bniq\b",
    r"\biri\b",
    r"\blocationgraph\b",
    r"\balesco\b",
    r"\b4\s*eyes\b",
    r"\baccountsgraph\b",
    r"\bdata alliance\b",
    # common wrapper / versioning / packaging buckets
    r"\boptimized\b",
    r"\bproscores\b",
    r"\bsurvey\b",
    r"\bv\s*\d+\b",       # "v 2" / "v2" when standalone token
    r"v\d+\b",             # catches "CPGv2" / "Intentv3" etc
]

_BAD_L1_EXACT_OR_CONTAINS = [
    "cookie", "cookies",
    "maid",
    "mobile id", "mobile ids",
    "ctv id", "ctv ids",
    "hashed email", "hashed emails",
    "id", "ids", "hem", "md5", "sha256", 
    "sha-256", "sha", "hash", "hashed",
]

# Exact L1s we know are vendor / product-line wrappers (keep this list small and grow as we observe)
_BAD_L1_EXACT = {
    "powerb2b",
    "media source solutions",
    "consumer watch network",
    "locationgraph",
    "accountsgraph.com",
}

def is_bad_l1_bucket(l1: str) -> bool:
    """
    Hard-block L1s that are clearly identifier/vendor/geo wrapper buckets.
    These should NEVER be taxonomy categories.
    """
    t = normalize_node(l1)
    tl = t.lower()
    if not tl:
        return True

    # geo / country-ish noise (common root junk)
    if tl in {"us", "usa", "ca", "uk", "au", "eu"}:
        return True

    # extended geo phrases
    if "united states" in tl:
        return True

    # obvious field-name wrappers (not real taxonomy categories)
    if tl in {"company name", "household id", "person id"}:
        return True

    # explicit known vendor wrappers
    if tl in _BAD_L1_EXACT:
        return True

    # vendor + geo style wrappers (e.g., "Acxiom US Health")
    # If it contains a geo token AND a known vendor token, drop even if it also contains category words like "health".
    if re.search(r"\b(us|usa|uk|au|eu)\b", tl) and re.search(r"\b(acxiom|exelate|nielsen|circana|niq|iri|locationgraph|alesco)\b", tl):
        return True

    # single-token mixed-case vendor wrappers (e.g., "PowerB2B", "LocationGraph")
    if re.fullmatch(r"[A-Za-z0-9]+", t) and not looks_like_category(t):
        # allow pure semantic acronyms like B2B
        if t.upper() not in {"B2B", "B2C"}:
            return True
    
    # dataset / brandline wrappers often look like "X by Y"
    if re.search(r"\bby\b", tl):
        return True

    if re.search(r"\b(md5|sha256|sha-256|hem|hash|hashed)\b", tl):
        return True

    # packaging / variant wrappers are not true categories
    # e.g., "Shopping & Retail(CPG)", "Optimized for TV", "CPGv2"
    if re.search(r"[()]", t):
        return True
    if "&" in t and "(" in t:
        return True
    if re.search(r"\boptimized\b", tl):
        return True
    if re.search(r"\bproscores\b", tl):
        return True
    if re.search(r"\bsurvey\b", tl):
        return True
    if re.search(r"\bv\s*\d+\b", tl):
        return True

    # very short ALLCAPS buckets that are usually wrappers (TVV/CPG/etc)
    # (allow B2B explicitly)
    if re.fullmatch(r"[A-Z]{2,4}", t.upper()) and t.upper() not in {"B2B"}:
        return True

    # short code-like buckets (e.g., "9D") are almost always wrappers/noise
    if re.fullmatch(r"[0-9]{1,2}[A-Za-z]{1,2}", t):
        return True

    # code-ish vendor wrappers that include B2B/B2C (e.g., "A1A B2B")
    if re.search(r"\d", t) and re.search(r"\b(b2b|b2c)\b", tl):
        return True

    # vendor-ish wrappers that include B2B/B2C but are not the pure category label
    # e.g., "PowerB2B", "SomethingB2C"
    if re.search(r"\b(b2b|b2c)\b", tl) and tl not in {"b2b", "b2c"}:
        # if it's a single token and contains digits, treat as wrapper
        if re.fullmatch(r"[a-z0-9]+", tl) and re.search(r"\d", tl):
            return True

    # version-suffixed buckets embedded in tokens (e.g., "CPGv2")
    if re.search(r"v\d+\b", tl):
        return True

    # contains exact/substring “ID-like” buckets
    for p in _BAD_L1_EXACT_OR_CONTAINS:
        if p in tl:
            return True

    # regex vendor/tech noise patterns
    for pat in _BAD_L1_PATTERNS:
        if re.search(pat, tl):
            return True

    return False

def is_reasonable_l1(l1: str) -> bool:
    """
    Allow new L1s, but block obvious vendor/tech-noise buckets.
    """
    t = normalize_node(l1)
    if not t:
        return False
    tl = t.lower()

    if len(t) > 40:
        return False

    if re.search(r"\d{2,}", t):
        return False

    for pat in _BAD_L1_PATTERNS:
        if re.search(pat, tl):
            return False

    if tl in {"us", "usa", "ca"}:
        return False

    return True


def normalize_l2_general(raw_l2: str) -> str:
    l2 = _lower(raw_l2)
    if not l2:
        return ""

    if "age" in l2:
        return "Age"
    if "income" in l2:
        return "Income"
    if "gender" in l2:
        return "Gender"
    if "household" in l2:
        return "Household"
    if "region" in l2 or "geo" in l2 or "geography" in l2:
        return "Region"
    if "lifecycle" in l2 or "lifestage" in l2:
        return "Lifecycle"

    if "functional area" in l2 or "job function" in l2:
        return "Functional Area"
    if "seniority" in l2 or "job seniority" in l2:
        return "Seniority"
    if "industry" in l2:
        return "Industry"
    if "role" in l2 or "job title" in l2 or "decision maker" in l2:
        return "Role"

    return normalize_node(raw_l2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="inp",
        default=DEFAULT_CATALOG_PATH,
        help="Input full catalog CSV",
    )
    ap.add_argument(
        "--out",
        dest="outp",
        default=DEFAULT_OUTPUT_PATH,
        help="Output training CSV",
    )
    ap.add_argument("--min-depth", type=int, default=3, help="Min nodes after provider (L1,L2,leaf => 3).")
    ap.add_argument(
        "--drop-abm",
        dest="drop_abm",
        action="store_true",
        help="Drop ABM/account-list patterns in path.",
    )
    ap.add_argument(
        "--no-drop-abm",
        dest="drop_abm",
        action="store_false",
        help="Do not drop ABM/account-list patterns in path.",
    )
    ap.add_argument("--keep-top-companies", action="store_true", help="Keep Top Companies trees.")
    ap.add_argument(
        "--gzip",
        dest="gzip",
        action="store_true",
        help="Write gzip-compressed CSV.",
    )
    ap.add_argument(
        "--no-gzip",
        dest="gzip",
        action="store_false",
        help="Write uncompressed CSV.",
    )
    ap.add_argument("--min-l1-count", type=int, default=20, help="Min per-provider frequency for an L1 to be accepted as L1 after shifts.")
    ap.add_argument("--max-dynamic-shifts", type=int, default=3, help="Shift forward until L1 looks category-ish or frequent enough.")
    ap.add_argument("--log-every", type=int, default=25000, help="Progress log interval. Set 0 to disable.")

    # Option B controls
    ap.add_argument(
        "--min-new-l1-count",
        type=int,
        default=200,
        help="Min per-provider frequency for a *novel* (non-category-ish) L1 to be kept.",
    )
    ap.add_argument(
        "--require-reasonable-new-l1",
        dest="require_reasonable_new_l1",
        action="store_true",
        help="Require novel L1s to pass is_reasonable_l1() (strict).",
    )
    ap.add_argument(
        "--no-require-reasonable-new-l1",
        dest="require_reasonable_new_l1",
        action="store_false",
        help="Allow novel L1s without is_reasonable_l1() strict check.",
    )

    args = ap.parse_args()

    # Default-on behavior so the script can be run with no args.
    ap.set_defaults(gzip=True)
    ap.set_defaults(drop_abm=True)
    ap.set_defaults(require_reasonable_new_l1=True)

    # Re-parse so the defaults above apply even if the flags were defined earlier.
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    drop_top_companies = not args.keep_top_companies

    in_path = Path(args.inp)
    out_path = Path(args.outp)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Reading input: %s", in_path)
    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
    logging.info("Loaded rows=%d cols=%d", len(df), len(df.columns))

    need = {"Provider Name", "Segment Name", "Segment Description"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    learned_roots_by_provider = learn_provider_roots(df)

    l1_freq_by_provider = _learn_l1_frequencies_after_root_shift(
        df,
        learned_roots_by_provider=learned_roots_by_provider,
        min_depth=args.min_depth,
    )

    rows: List[Dict[str, Any]] = []

    processed = 0
    kept = 0
    dropped = 0
    dropped_new_l1 = 0

    for _, r in df.iterrows():
        processed += 1
        if args.log_every and args.log_every > 0 and processed % args.log_every == 0:
            logging.info(
                "Progress: processed=%d kept=%d dropped=%d dropped_new_l1=%d",
                processed, kept, dropped, dropped_new_l1
            )

        provider = normalize_node(r.get("Provider Name", ""))
        seg = r.get("Segment Name", "")
        parts = split_parts(seg)

        if not provider or not parts:
            dropped += 1
            continue

        # Drop provider token if present
        if parts and _lower(parts[0]) == _lower(provider):
            parts = parts[1:]

        if len(parts) < args.min_depth:
            dropped += 1
            continue

        # Root shifts
        learned_roots = learned_roots_by_provider.get(provider, set())
        parts = _apply_root_shifts(parts, provider, learned_roots, max_shifts=2)

        if len(parts) < args.min_depth:
            dropped += 1
            continue

        # Dynamic shift until category-ish or frequent enough
        provider_freq = l1_freq_by_provider.get(provider, {})
        for _ in range(max(0, args.max_dynamic_shifts)):
            if len(parts) < args.min_depth:
                break

            cand_l1 = normalize_node(parts[0])
            cand_count = provider_freq.get(cand_l1, 0)

            if looks_like_category(cand_l1) or cand_count >= args.min_l1_count:
                break

            parts = parts[1:]

        if len(parts) < args.min_depth:
            dropped += 1
            continue

        l1 = normalize_node(parts[0])
        l2 = normalize_node(parts[1])
        leaf = normalize_node(parts[-1])

        if not l1 or not l2 or not leaf:
            dropped += 1
            continue

        # Drop ABM/top-companies junk
        if should_drop_tree(parts, drop_abm=args.drop_abm, drop_top_companies=drop_top_companies):
            dropped += 1
            continue
        # hard block obvious junk L1 buckets (IDs/cookies/etc)
        if is_bad_l1_bucket(l1):
            dropped += 1
            dropped_new_l1 += 1
            continue

        # ✅ Option B gating for NEW L1s:
        # - Category-ish L1s: allow (even if not huge frequency)
        # - Non-category-ish L1s: require either:
        #     (a) frequent enough to be stable (min_l1_count) OR
        #     (b) "novel but real": >= min_new_l1_count and (optionally) reasonable
        l1_count = provider_freq.get(l1, 0)


        if not looks_like_category(l1):
            keep = False

            if l1_count >= args.min_l1_count:
                keep = True
            elif l1_count >= args.min_new_l1_count:
                if args.require_reasonable_new_l1:
                    keep = is_reasonable_l1(l1)
                else:
                    # non-strict: keep frequent novel L1s even if a bit weird,
                    # but still drop the worst offenders
                    keep = is_reasonable_l1(l1)

            if not keep:
                dropped += 1
                dropped_new_l1 += 1
                continue

        l2_norm = normalize_l2_general(l2)
        if not l2_norm:
            dropped += 1
            continue

        full_path = "Cybba > " + " > ".join([l1, l2_norm, leaf])

        name = clean(r.get("Segment Name", ""))
        desc = clean(r.get("Segment Description", ""))
        field = clean(r.get("Field Name", ""))
        value = clean(r.get("Value Name", ""))

        text = " | ".join([x for x in [name, desc, field, value] if x])

        rows.append(
            {
                "text": text,
                "L1": l1,
                "L2": l2_norm,
                "Leaf": leaf,
                "full_path": full_path,
                "Provider Name": provider,
                "Segment Name": name,
            }
        )
        kept += 1

    out = pd.DataFrame(rows)

    if args.gzip:
        out.to_csv(out_path, index=False, encoding="utf-8", compression="gzip")
    else:
        out.to_csv(out_path, index=False, encoding="utf-8")

    logging.info(
        "Done. Wrote %d rows -> %s (dropped=%d dropped_new_l1=%d)",
        len(out), out_path, dropped, dropped_new_l1
    )
    print(f"Wrote {len(out)} training rows -> {out_path}")


if __name__ == "__main__":
    main()