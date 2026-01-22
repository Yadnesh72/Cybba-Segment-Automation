# backend/src/validators.py
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd


# ----------------------------
# Normalization + parsing
# ----------------------------

def clean(s: str) -> str:
    return str(s or "").strip()


def normalize_name(s: str) -> str:
    """
    Normalizes a segment name for net-new checks:
    - lowercase
    - remove punctuation
    - collapse whitespace
    """
    s = clean(s).lower()
    s = re.sub(r"[^\w\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_component_ids(components_str: str) -> List[str]:
    """
    Parse "Non Derived Segments utilized" field produced by the model:
      "Field:Value | Segment Name; Field:Value | Segment Name; ..."
    Returns:
      ["Field:Value", ...]
    """
    out: List[str] = []
    s = clean(components_str)
    if not s:
        return out

    parts = [p.strip() for p in s.split(";") if p.strip()]
    for p in parts:
        cid = p.split("|", 1)[0].strip()
        if cid:
            out.append(cid)
    return out


def stable_hash_suffix(component_ids: Iterable[str], n: int = 8) -> str:
    """
    Deterministic suffix based on sorted component IDs.
    """
    key = "|".join(sorted(map(str, component_ids))).encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:n]


# ----------------------------
# Naming convention enforcement
# ----------------------------

_ACRONYMS = {"B2B", "ABM", "TV", "CPM", "USA", "US", "UK", "AFOL"}


def split_name_parts(name: str, sep: str) -> List[str]:
    """
    Robust split on '>' even if spacing varies.
    """
    s = clean(name)
    if not s:
        return []
    if ">" in sep:
        parts = re.split(r"\s*>\s*", s)
    else:
        parts = s.split(sep)
    return [p.strip() for p in parts if p and p.strip()]


def join_name_parts(parts: List[str]) -> str:
    parts = [clean(p) for p in parts if clean(p)]
    return " > ".join(parts)


def title_case_node(node: str) -> str:
    t = clean(node)
    if not t:
        return t
    if t.upper() in _ACRONYMS:
        return t.upper()
    return t.title()


def remove_trailing_deterministic(s: str) -> str:
    """
    Remove " - Deterministic" suffixes from leaf/tail.
    """
    t = clean(s)
    return re.sub(r"\s*-\s*deterministic\s*$", "", t, flags=re.IGNORECASE).strip()


def looks_like_hash_or_id_token(s: str) -> bool:
    """
    Reject leaf fragments that look like:
      - '0 - 656fedbb'
      - '656fedbb'
      - long hex tokens
      - pure numbers
    """
    t = clean(s)
    if not t:
        return True
    if re.fullmatch(r"\d+", t):
        return True
    if re.fullmatch(r"[a-f0-9]{8,}", t.lower()):
        return True
    # pattern like "0 - deadbeef"
    if re.fullmatch(r"\d+\s*-\s*[a-f0-9]{6,}", t.lower()):
        return True
    return False


def extract_cybba_l1_l2_leaf(name: str, provider: str, sep: str) -> Tuple[str, str, str]:
    """
    Normalize any Proposed New Segment Name into:
      Cybba > L1 > L2 > Leaf

    Behavior:
    - Ensures provider is first; if not, return ("","","")
    - Removes duplicate provider tokens: "Cybba > Cybba > ..."
    - If there are more than 4 nodes, compress tail into leaf with " - "
    - If fewer than 4, return ("","","")
    """
    parts = split_name_parts(name, sep)
    if not parts:
        return "", "", ""

    if parts[0].lower() != provider.lower():
        return "", "", ""

    # remove duplicate provider tokens
    while len(parts) >= 2 and parts[1].lower() == provider.lower():
        parts.pop(1)

    if len(parts) < 4:
        return "", "", ""

    if len(parts) > 4:
        head = parts[:3]          # provider, L1, L2
        tail = parts[3:]          # leaf + extra
        leaf = " - ".join([clean(x) for x in tail if clean(x)])
        parts = head + [leaf]

    provider_tok = clean(parts[0])
    l1 = clean(parts[1])
    l2 = clean(parts[2])
    leaf = clean(parts[3])

    return provider_tok, l1, l2, leaf  # type: ignore[return-value]


def enforce_clean_segment_names(
    proposals_df: pd.DataFrame,
    cfg: dict,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Enforce the *format* the business needs:

      Cybba > L1 > L2 > Leaf

    IMPORTANT: We normalize/fix instead of dropping most rows.
    We only drop when:
      - provider isn't first
      - can't get to 4 nodes
      - leaf is empty/junk/id/hash
    """
    logger = logger or logging.getLogger(__name__)
    require_columns(proposals_df, ["Proposed New Segment Name"], context="proposals_df")

    provider = clean(cfg["provider"]["cybba_name"])
    sep = clean(cfg.get("taxonomy", {}).get("separator", " > "))

    kept_rows: List[pd.Series] = []
    dropped = 0
    rewritten = 0

    for _, row in proposals_df.iterrows():
        original = clean(row["Proposed New Segment Name"])
        p, l1, l2, leaf = extract_cybba_l1_l2_leaf(original, provider=provider, sep=sep)
        if not p:
            dropped += 1
            continue

        # normalize taxonomy nodes
        l1 = title_case_node(l1)
        l2 = title_case_node(l2)

        # leaf cleanup
        leaf = remove_trailing_deterministic(leaf)
        leaf = leaf.strip(" -").strip()
        leaf = title_case_node(leaf)

        if looks_like_hash_or_id_token(leaf):
            dropped += 1
            continue

        fixed = join_name_parts([provider, l1, l2, leaf])

        new_row = row.copy()
        new_row["Proposed New Segment Name"] = fixed
        kept_rows.append(new_row)

        if fixed != original:
            rewritten += 1

    out = pd.DataFrame(kept_rows)
    logger.info(
        "Naming convention enforcement: kept=%d dropped=%d rewritten=%d",
        len(out), dropped, rewritten
    )
    return out, dropped


# ----------------------------
# Data universe builders
# ----------------------------

def build_underived_id_universe(
    underived_df: pd.DataFrame,
    cfg: dict,
) -> Set[str]:
    """
    Builds the set of allowed component IDs from Input B (underived).
    Default strategy: "field_value" => FieldID:ValueID
    """
    strat = cfg.get("underived_id", {}).get("strategy", "field_value")

    if strat == "field_value":
        fcol = cfg["underived_id"]["field_id_col"]
        vcol = cfg["underived_id"]["value_id_col"]

        missing = [c for c in [fcol, vcol] if c not in underived_df.columns]
        if missing:
            raise ValueError(
                f"Underived Input B missing required columns for underived_id.strategy=field_value: {missing}"
            )

        f = underived_df[fcol].astype(str).str.strip()
        v = underived_df[vcol].astype(str).str.strip()
        ids = (f + ":" + v).tolist()
        return {x for x in ids if x and x != ":"}

    raise ValueError(f"Unsupported underived_id.strategy: {strat}")


def build_existing_cybba_name_set(
    distributed_df: pd.DataFrame,
    cfg: dict,
) -> Set[str]:
    """
    Builds a set of normalized names for existing Cybba distributed segments in Input A.
    Used for net-new enforcement by name.
    """
    cybba_provider = cfg["provider"]["cybba_name"].lower()

    if "Provider Name" not in distributed_df.columns or "Segment Name" not in distributed_df.columns:
        raise ValueError("Input A must contain 'Provider Name' and 'Segment Name' columns to build existing set.")

    cybba = distributed_df[distributed_df["Provider Name"].astype(str).str.lower() == cybba_provider]
    return set(cybba["Segment Name"].astype(str).map(normalize_name))


# ----------------------------
# Validators
# ----------------------------

@dataclass
class ValidationReport:
    input_rows: int
    after_naming: int
    after_underived_only: int
    after_net_new: int
    after_collision_resolution: int
    dropped_naming: int
    dropped_underived_only: int
    dropped_net_new: int
    collisions_resolved: int


def require_columns(df: pd.DataFrame, cols: List[str], context: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} {('in ' + context) if context else ''}")


def enforce_underived_only(
    proposals_df: pd.DataFrame,
    underived_id_universe: Set[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, int]:
    logger = logger or logging.getLogger(__name__)
    require_columns(proposals_df, ["Non Derived Segments utilized"], context="proposals_df")

    keep_idx = []
    dropped = 0

    for i, row in proposals_df.iterrows():
        ids = parse_component_ids(row["Non Derived Segments utilized"])
        if not ids:
            dropped += 1
            continue
        if all(cid in underived_id_universe for cid in ids):
            keep_idx.append(i)
        else:
            dropped += 1

    out = proposals_df.loc[keep_idx].copy()
    logger.info("Underived-only enforcement: kept=%d dropped=%d", len(out), dropped)
    return out, dropped


def enforce_net_new_by_name(
    proposals_df: pd.DataFrame,
    existing_cybba_names_norm: Set[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, int]:
    logger = logger or logging.getLogger(__name__)
    require_columns(proposals_df, ["Proposed New Segment Name"], context="proposals_df")

    keep_idx = []
    dropped = 0

    for i, row in proposals_df.iterrows():
        nn = normalize_name(row["Proposed New Segment Name"])
        if nn in existing_cybba_names_norm:
            dropped += 1
        else:
            keep_idx.append(i)

    out = proposals_df.loc[keep_idx].copy()
    logger.info("Net-new by name: kept=%d dropped=%d", len(out), dropped)
    return out, dropped


def resolve_name_collisions(
    proposals_df: pd.DataFrame,
    cfg: dict,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, int]:
    logger = logger or logging.getLogger(__name__)
    require_columns(proposals_df, ["Proposed New Segment Name", "Non Derived Segments utilized"], context="proposals_df")

    enabled = bool(cfg.get("dedupe", {}).get("resolve_name_collision", True))
    if not enabled:
        return proposals_df.copy(), 0

    hash_len = int(cfg.get("dedupe", {}).get("collision_hash_len", 8))

    seen = {}
    new_names = []
    collisions = 0

    for _, row in proposals_df.iterrows():
        name = row["Proposed New Segment Name"]
        ids = parse_component_ids(row["Non Derived Segments utilized"])

        if name not in seen:
            seen[name] = 1
            new_names.append(name)
            continue

        collisions += 1
        suffix = stable_hash_suffix(ids, n=hash_len)
        adjusted = f"{name} - {suffix}"
        logger.warning("Name collision: '%s' adjusted to '%s'", name, adjusted)
        new_names.append(adjusted)

    out = proposals_df.copy()
    out["Proposed New Segment Name"] = new_names
    logger.info("Collision resolution: collisions_resolved=%d", collisions)
    return out, collisions


def validate_and_prepare(
    proposals_df: pd.DataFrame,
    *,
    underived_df: pd.DataFrame,
    distributed_df: pd.DataFrame,
    cfg: dict,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Runs the validator suite in order:
    0) naming convention enforcement (normalize/fix instead of dropping)
    1) underived-only enforcement
    2) net-new enforcement
    3) deterministic name collision resolution
    """
    logger = logger or logging.getLogger(__name__)

    require_columns(proposals_df, ["Proposed New Segment Name", "Non Derived Segments utilized"], context="proposals_df")

    report = ValidationReport(
        input_rows=len(proposals_df),
        after_naming=len(proposals_df),
        after_underived_only=len(proposals_df),
        after_net_new=len(proposals_df),
        after_collision_resolution=len(proposals_df),
        dropped_naming=0,
        dropped_underived_only=0,
        dropped_net_new=0,
        collisions_resolved=0,
    )

    # 0) Naming convention enforcement
    df0, dropped_name = enforce_clean_segment_names(proposals_df, cfg, logger=logger)
    report.after_naming = len(df0)
    report.dropped_naming = dropped_name

    # Build universes
    underived_universe = build_underived_id_universe(underived_df, cfg)
    existing_cybba_names = build_existing_cybba_name_set(distributed_df, cfg)

    # 1) Underived-only
    df1, dropped_u = enforce_underived_only(df0, underived_universe, logger=logger)
    report.after_underived_only = len(df1)
    report.dropped_underived_only = dropped_u

    # 2) Net-new
    df2, dropped_n = enforce_net_new_by_name(df1, existing_cybba_names, logger=logger)
    report.after_net_new = len(df2)
    report.dropped_net_new = dropped_n

    # 3) Name collision resolution
    df3, collisions = resolve_name_collisions(df2, cfg, logger=logger)
    report.after_collision_resolution = len(df3)
    report.collisions_resolved = collisions

    return df3, report
