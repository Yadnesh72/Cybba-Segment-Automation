# backend/src/web_assistance.py
"""
LLM-Assisted Segment Generation
================================
When enabled, this module:
  1. Loads the actual Cybba catalog and competitor catalog from disk.
  2. Runs a gap analysis to find category areas that competitors cover
     but Cybba either misses entirely or has very thin coverage on.
  3. Passes those gaps + existing context to Ollama to generate targeted
     new Cybba-style segments that fill the real market opportunities.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("web_assistance")

# ── Output schema ──────────────────────────────────────────────────────────────
_PROPOSAL_COLS = [
    "Competitor Provider",
    "Competitor Segment Name",
    "Proposed New Segment Name",
    "Non Derived Segments utilized",
    "Composition Similarity",
    "Closest Cybba Segment",
    "Closest Cybba Similarity",
    "L1",
    "L2",
    "Leaf",
    "web_assisted",
]

# ── Trend themes used as supplementary inspiration when gaps are too few ───────
_TREND_THEMES = [
    "Biohackers: CGM users, cold-plunge enthusiasts, nootropic buyers, longevity clinic patients",
    "Creator Economy: Substack writers, course creators, podcast hosts, Patreon supporters",
    "AI Tool Power Users: ChatGPT Pro subscribers, AI writing tool daily users, prompt engineers",
    "Women's Health: menopause support seekers, fertility tracking users, PCOS management",
    "EV Ecosystem: EV owners, home charging station buyers, fleet electrification decision makers",
    "Silver Economy: tech-savvy retirees, active seniors 65+, age-in-place home modifiers",
    "No-Code Builders: Webflow designers, Bubble developers, Zapier automation builders",
    "Mental Health Tech: therapy platform users, mood tracking app users, burnout recovery seekers",
    "Sustainable Living: zero-waste adopters, carbon offset purchasers, sustainable fashion buyers",
    "Buy Now Pay Later: BNPL frequent shoppers, Klarna/Afterpay users, young credit-builders",
    "Digital Nomads: remote work travelers, co-living space members, long-stay Airbnb bookers",
    "Legal Operations: legal ops managers, e-discovery tool users, contract management buyers",
    "Pickleball Players: equipment buyers, court booking users, recreational sports participants",
    "Urban Farming: hydroponic gardeners, backyard chicken owners, mushroom cultivation hobbyists",
    "Sleep Tech: smart mattress owners, sleep tracking users, white noise machine buyers",
    "Construction & Trades: general contractors, residential remodelers, specialty subcontractors",
    "Regenerative Tourism: eco-lodge bookers, voluntourism participants, wildlife conservation travelers",
    "Embedded Finance: neobank customers, payroll advance app users, super-app financial users",
    "Fantasy Sports & Betting: daily fantasy participants, sports betting platform users",
    "Longevity Seekers: anti-aging supplement buyers, NAD+ purchasers, biological age test takers",
]


# ── Catalog loading ────────────────────────────────────────────────────────────

def _load_full_catalog(cfg: dict) -> pd.DataFrame:
    """Load input_a — the full marketplace catalog (all providers including Cybba)."""
    try:
        input_dir = Path(cfg["paths"]["input_dir"])
        df = pd.read_csv(
            input_dir / cfg["files"]["input_a"],
            dtype=str,
            keep_default_na=False,
            usecols=lambda c: c in ("Provider Name", "Segment Name", "Segment Description"),
        )
        return df
    except Exception as e:
        logger.warning("Could not load full catalog: %s", e)
        return pd.DataFrame()


# ── Name parsing ───────────────────────────────────────────────────────────────

def _parts(name: str) -> List[str]:
    """Split 'A > B > C > D' into ['A','B','C','D'], stripped."""
    return [p.strip().lstrip("*").strip() for p in str(name).split(">") if p.strip()]


# ── Gap analysis ───────────────────────────────────────────────────────────────

_TOP_PROVIDERS = {
    "Experian", "TransUnion", "Acxiom", "Oracle Data Cloud", "Neustar",
    "IRI", "LiveRamp", "Dun & Bradstreet", "Bombora", "Epsilon",
    "Nielsen", "Polk", "comScore", "V12 Group", "TruAudience",
    "Verisk", "Kochava", "Lotame",
}


def _find_gaps(full_catalog: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Compare Cybba L2 coverage against top-provider competitor L2 coverage.

    Returns a list of gap dicts, priority-sorted:
      missing_l2  — competitor has ≥3 segments in this L2; Cybba has 0
      thin_l2     — competitor has ≥5 segments; Cybba has <2
      sparse_l1   — competitor has ≥10 segments in L1; Cybba <25% coverage
    """
    if full_catalog.empty or "Segment Name" not in full_catalog.columns:
        return []

    cybba_mask = full_catalog.get("Provider Name", pd.Series(dtype=str)) == "Cybba"
    cybba_df   = full_catalog[cybba_mask]
    comp_df    = full_catalog[~cybba_mask]

    # Optionally narrow to well-structured providers (skip if too few)
    structured = comp_df[comp_df["Provider Name"].isin(_TOP_PROVIDERS)]
    if len(structured) < 500:
        structured = comp_df  # fall back to all

    # ── Parse Cybba L1/L2 ─────────────────────────────────────────────────────
    # Cybba names: "Cybba > L1 > L2 > Leaf"  (parts[0]='Cybba', parts[1]=L1, parts[2]=L2)
    cybba_l1: Counter = Counter()
    cybba_l2: Counter = Counter()
    for name in cybba_df["Segment Name"].dropna():
        p = _parts(name)
        # Skip the "Cybba" prefix part
        cats = p[1:] if (p and p[0].lower() == "cybba") else p
        l1 = cats[0] if len(cats) > 0 else ""
        l2 = cats[1] if len(cats) > 1 else ""
        if l1:
            cybba_l1[l1] += 1
        if l1 and l2:
            cybba_l2[f"{l1} > {l2}"] += 1

    # ── Parse competitor L1/L2 ────────────────────────────────────────────────
    # Competitor names: "Provider > L1 > L2 > Leaf" (parts[0]=provider, parts[1]=L1 ...)
    comp_l1: Counter = Counter()
    comp_l2: Counter = Counter()
    comp_l2_examples: Dict[str, List[str]] = defaultdict(list)

    for row in structured.itertuples(index=False):
        name = str(row[structured.columns.get_loc("Segment Name")] if hasattr(row, "_fields") else "")
        p = _parts(name)
        # Skip provider name (first token) — it often matches Provider Name col
        cats = p[1:] if len(p) > 1 else p
        l1 = cats[0] if len(cats) > 0 else ""
        l2 = cats[1] if len(cats) > 1 else ""
        if l1:
            comp_l1[l1] += 1
        if l1 and l2:
            key = f"{l1} > {l2}"
            comp_l2[key] += 1
            if len(comp_l2_examples[key]) < 4:
                comp_l2_examples[key].append(str(p[-1]) if p else name)

    gaps: List[Dict[str, Any]] = []
    seen: set = set()

    # Gap type 1 — L2 entirely absent from Cybba (comp ≥ 3)
    for key, cnt in comp_l2.most_common(200):
        if cnt < 3:
            continue
        l1, _, l2 = key.partition(" > ")
        # Skip if this L2 is actually a superset/prefix match of a Cybba L2
        if cybba_l2.get(key, 0) > 0:
            continue
        tag = (l1, l2)
        if tag not in seen:
            seen.add(tag)
            gaps.append({
                "type": "missing_l2",
                "l1": l1, "l2": l2,
                "competitor_count": cnt,
                "cybba_count": 0,
                "examples": comp_l2_examples[key][:3],
            })

    # Gap type 2 — L2 very thin in Cybba (comp ≥ 5, Cybba < 2)
    for key, cnt in comp_l2.most_common(200):
        if cnt < 5:
            continue
        l1, _, l2 = key.partition(" > ")
        cybba_cnt = cybba_l2.get(key, 0)
        if cybba_cnt >= 2:
            continue
        tag = (l1, l2)
        if tag not in seen:
            seen.add(tag)
            gaps.append({
                "type": "thin_l2",
                "l1": l1, "l2": l2,
                "competitor_count": cnt,
                "cybba_count": cybba_cnt,
                "examples": comp_l2_examples[key][:3],
            })

    # Gap type 3 — Entire L1 sparse in Cybba (comp ≥ 10, Cybba < 25%)
    for l1, cnt in comp_l1.most_common(50):
        if cnt < 10:
            continue
        cybba_cnt = cybba_l1.get(l1, 0)
        if cybba_cnt >= cnt * 0.25:
            continue
        tag = (l1, "")
        if tag not in seen:
            seen.add(tag)
            gaps.append({
                "type": "sparse_l1",
                "l1": l1, "l2": "",
                "competitor_count": cnt,
                "cybba_count": cybba_cnt,
                "examples": [],
            })

    # Sort: missing > thin > sparse, then by competitor volume descending
    _prio = {"missing_l2": 0, "thin_l2": 1, "sparse_l1": 2}
    gaps.sort(key=lambda g: (_prio.get(g["type"], 9), -g["competitor_count"]))

    logger.info(
        "Gap analysis: %d missing_l2, %d thin_l2, %d sparse_l1 gaps found",
        sum(1 for g in gaps if g["type"] == "missing_l2"),
        sum(1 for g in gaps if g["type"] == "thin_l2"),
        sum(1 for g in gaps if g["type"] == "sparse_l1"),
    )
    return gaps[:35]


# ── Prompt builder — one focused prompt per gap ────────────────────────────────

def _build_focused_prompt(
    gap: Dict[str, Any],
    cybba_sample: List[str],
    already_generated: List[str],
    n_segments: int,
) -> str:
    """Build a tight prompt for a single specific gap."""
    l1 = gap["l1"]
    l2 = gap.get("l2", "")
    category = f"{l1} > {l2}" if l2 else l1

    examples_text = ""
    if gap.get("examples"):
        examples_text = (
            f"\nCompetitor leaf examples in this category: "
            + ", ".join(f'"{e}"' for e in gap["examples"][:4])
        )

    avoid_lines = (already_generated + cybba_sample)[-40:]
    avoid_text = "\n".join(f"  - {n}" for n in avoid_lines) if avoid_lines else "  (none yet)"

    name_example = f"{l1} > {l2} > Specific Leaf" if l2 else f"{l1} > Subcategory > Specific Leaf"

    return f"""You are a digital advertising audience segment specialist.

TASK
Generate exactly {n_segments} brand-new audience segment(s) for this specific category:
  {category}

GAP CONTEXT
  Type: {gap['type'].replace('_', ' ').upper()}
  Competitors have {gap['competitor_count']} segments here; Cybba has only {gap['cybba_count']}.{examples_text}

NAMING RULES
  - Format: {name_example}
  - L1 must be exactly: {l1}
  - L2 must be exactly: {l2 if l2 else '<pick a relevant subcategory>'}
  - Leaf = 3–6 words, specific, advertiser-actionable audience descriptor
  - Good leaf: "DIFM Oil Change Intenders", "Luxury EV Considerers", "Gluten-Free CPG Shoppers"
  - Bad leaf: "Auto People", "Health Users", "Finance Segment"
  - Think niches, life stages, purchase intenders, behavioral patterns within this category.
  - Each of the {n_segments} segments must have a DIFFERENT leaf name — no variations of the same idea.

DO NOT GENERATE ANY OF THESE (already exists):
{avoid_text}

Output ONLY a valid JSON array — no markdown, no prose, no explanation.
[
  {{"name": "{name_example}", "l1": "{l1}", "l2": "{l2 if l2 else 'subcategory'}", "leaf": "Leaf Name Here"}},
  ...
]
Generate exactly {n_segments} items now:"""


def _build_trend_prompt(
    theme: str,
    cybba_sample: List[str],
    already_generated: List[str],
    n_segments: int,
) -> str:
    """Fallback prompt for trend-based segments when gaps are exhausted."""
    avoid_lines = (already_generated + cybba_sample)[-40:]
    avoid_text = "\n".join(f"  - {n}" for n in avoid_lines) if avoid_lines else "  (none yet)"

    return f"""You are a digital advertising audience segment specialist.

TASK
Generate exactly {n_segments} new audience segment(s) inspired by this emerging trend:
  {theme}

NAMING RULES
  - Format: L1 > L2 > Leaf  (e.g. "Health & Wellness > Fitness > Wearable Fitness Tracker Users")
  - Leaf = 3–6 words, specific, advertiser-actionable audience descriptor
  - Each segment must target a DIFFERENT niche within the trend

DO NOT GENERATE ANY OF THESE (already exists):
{avoid_text}

Output ONLY a valid JSON array — no markdown, no prose.
[
  {{"name": "L1 > L2 > Leaf", "l1": "L1", "l2": "L2", "leaf": "Leaf Name Here"}},
  ...
]
Generate exactly {n_segments} items now:"""


# ── LLM callers ────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, ollama_url: str, model: str, timeout: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 2048},
    }
    resp = requests.post(ollama_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return str(resp.json().get("response", "")).strip()


def _extract_json_list(text: str) -> List[Dict[str, Any]]:
    # Try first JSON array found in text
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return []


# ── Novelty dedup ──────────────────────────────────────────────────────────────

def _novel_mask(names: List[str], existing: List[str], threshold: float = 0.45) -> List[bool]:
    """True = novel enough to keep."""
    if not existing or not names:
        return [True] * len(names)
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        vec.fit(existing + names)
        sims = cosine_similarity(vec.transform(names), vec.transform(existing))
        return [float(sims[i].max()) < threshold for i in range(len(names))]
    except Exception as e:
        logger.warning("Novelty filter error: %s", e)
        return [True] * len(names)


# ── Per-call item parser ────────────────────────────────────────────────────────

_PLACEHOLDER_LEAVES = {
    "specific leaf", "leaf name here", "leaf", "subcategory", "segment",
    "audience segment", "new segment", "placeholder", "tbd", "example leaf",
}

def _parse_items(raw: str) -> List[Dict[str, str]]:
    items = _extract_json_list(raw)
    candidates: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        l1   = str(item.get("l1")   or "").strip()
        l2   = str(item.get("l2")   or "").strip()
        leaf = str(item.get("leaf") or "").strip()
        if not name and (l1 or l2 or leaf):
            name = " > ".join(p for p in [l1, l2, leaf] if p)
        if len(name) < 5:
            continue
        # Skip items where the leaf is still the template placeholder
        if leaf.lower() in _PLACEHOLDER_LEAVES:
            continue
        # Skip if the full name ends with a placeholder leaf
        last_part = name.split(">")[-1].strip().lower()
        if last_part in _PLACEHOLDER_LEAVES:
            continue
        candidates.append({"name": name, "l1": l1, "l2": l2, "leaf": leaf})
    return candidates


# ── Main public function ───────────────────────────────────────────────────────

def generate_web_assisted_segments(
    cfg: dict,
    existing_catalog_names: List[str],
    *,
    ollama_url: str = "http://host.docker.internal:11434/api/generate",
    model: str = "llama3.1",
    max_segments: int = 20,
    max_search_queries: int = 5,   # kept for API compat — unused
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Performs real gap analysis on the Cybba + competitor catalogs, then
    makes one focused Ollama call per gap to generate targeted segments.

    Running one call per gap (instead of one big call) ensures:
      - The LLM stays focused on a single category per call
      - Already-generated names are excluded from each subsequent call
      - Output is diverse across many different L1/L2 categories
    """
    # Segments per gap call — 2 keeps calls fast and focused
    SEGS_PER_GAP = 2

    logger.info("LLM web assistance: starting per-gap generation (max=%d)", max_segments)

    try:
        # ── 1. Load full catalog ──────────────────────────────────────────────
        full_catalog = _load_full_catalog(cfg)

        # ── 2. Gap analysis ───────────────────────────────────────────────────
        if not full_catalog.empty:
            gaps = _find_gaps(full_catalog)
        else:
            gaps = []
            logger.warning("LLM web assistance: catalog not loaded — using trend themes only")

        # ── 3. Budget ─────────────────────────────────────────────────────────
        # Iterate through gaps until max_segments is reached; trend calls fill any remainder.

        # ── 4. Filter trend themes to novel ones ──────────────────────────────
        if existing_catalog_names:
            try:
                vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
                vec.fit(existing_catalog_names + _TREND_THEMES)
                sims = cosine_similarity(
                    vec.transform(_TREND_THEMES),
                    vec.transform(existing_catalog_names),
                )
                novel_trends = [t for i, t in enumerate(_TREND_THEMES) if float(sims[i].max()) < 0.38]
                if len(novel_trends) < 5:
                    novel_trends = _TREND_THEMES
            except Exception:
                novel_trends = _TREND_THEMES
        else:
            novel_trends = _TREND_THEMES

        cybba_sample = existing_catalog_names[:50]

        # ── 5. One Ollama call per gap ────────────────────────────────────────
        records: List[Dict] = []
        already_generated: List[str] = []   # grows with each call — prevents cross-call dupes

        logger.info(
            "LLM web assistance: %d gaps available, budget=%d, using Ollama (%s)",
            len(gaps), max_segments, model,
        )

        for gap in gaps:
            if len(records) >= max_segments:
                break
            category_label = f"{gap['l1']} > {gap['l2']}" if gap['l2'] else gap['l1']
            try:
                prompt = _build_focused_prompt(
                    gap=gap,
                    cybba_sample=cybba_sample,
                    already_generated=already_generated,
                    n_segments=SEGS_PER_GAP,
                )
                raw = _call_ollama(prompt, ollama_url=ollama_url, model=model, timeout=timeout)
                candidates = _parse_items(raw)

                novel = _novel_mask(
                    [c["name"] for c in candidates],
                    existing_catalog_names + already_generated,
                )
                added = 0
                for flag, c in zip(novel, candidates):
                    if not flag:
                        logger.debug("filtered near-duplicate: %s", c["name"])
                        continue
                    records.append({
                        "Competitor Provider":           "LLM Assisted",
                        "Competitor Segment Name":       f"{c['l2']} > {c['leaf']}" if (c["l2"] and c["leaf"]) else c["name"],
                        "Proposed New Segment Name":     c["name"],
                        "Non Derived Segments utilized": "",
                        "Composition Similarity":        0.0,
                        "Closest Cybba Segment":         "",
                        "Closest Cybba Similarity":      0.0,
                        "L1":          c["l1"],
                        "L2":          c["l2"],
                        "Leaf":        c["leaf"],
                        "web_assisted": True,
                    })
                    already_generated.append(c["name"])
                    added += 1

                logger.info("gap [%s]: +%d segments", category_label, added)

            except Exception as e:
                logger.warning("gap [%s]: call failed (%s), skipping", category_label, e)
                continue

        # ── 6. Trend calls to top up if gaps didn't fill the budget ──────────
        for theme in novel_trends:
            if len(records) >= max_segments:
                break
            try:
                prompt = _build_trend_prompt(
                    theme=theme,
                    cybba_sample=cybba_sample,
                    already_generated=already_generated,
                    n_segments=SEGS_PER_GAP,
                )
                raw = _call_ollama(prompt, ollama_url=ollama_url, model=model, timeout=timeout)
                candidates = _parse_items(raw)

                novel = _novel_mask(
                    [c["name"] for c in candidates],
                    existing_catalog_names + already_generated,
                )
                for flag, c in zip(novel, candidates):
                    if not flag:
                        continue
                    records.append({
                        "Competitor Provider":           "LLM Assisted",
                        "Competitor Segment Name":       f"{c['l2']} > {c['leaf']}" if (c["l2"] and c["leaf"]) else c["name"],
                        "Proposed New Segment Name":     c["name"],
                        "Non Derived Segments utilized": "",
                        "Composition Similarity":        0.0,
                        "Closest Cybba Segment":         "",
                        "Closest Cybba Similarity":      0.0,
                        "L1":          c["l1"],
                        "L2":          c["l2"],
                        "Leaf":        c["leaf"],
                        "web_assisted": True,
                    })
                    already_generated.append(c["name"])

            except Exception as e:
                logger.warning("trend call failed (%s), skipping", e)
                continue

        if not records:
            logger.warning("LLM web assistance: all candidates filtered or no calls succeeded")
            return pd.DataFrame(columns=_PROPOSAL_COLS)

        df = pd.DataFrame(records)
        for col in _PROPOSAL_COLS:
            if col not in df.columns:
                df[col] = 0.0 if col in ("Composition Similarity", "Closest Cybba Similarity") else ""

        logger.info("LLM web assistance: returning %d diverse gap-targeted proposals", len(df))
        return df

    except Exception as e:
        logger.error("LLM web assistance: unexpected error: %s", e, exc_info=True)
        return pd.DataFrame(columns=_PROPOSAL_COLS)
