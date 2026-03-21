#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from pathlib import Path
import pandas as pd
import os
import time

# ------------------------------------------------------------
# Defaults (run with no args)
# ------------------------------------------------------------
DEFAULT_CONFIG_PATH = "/Users/yadnesh/cybba/cybba_segment_automation/backend/config/config.yml"
DEFAULT_DB_PATH = "/Users/yadnesh/cybba/cybba_segment_automation/backend/data/runs.db"

# Rolling append output (kept in same folder as other raw segments by default)
DEFAULT_APPEND_DIR = "/Users/yadnesh/cybba/cybba_segment_automation/Data/Input/Raw_segments/training_append"
DEFAULT_OUT_PATH = "/Users/yadnesh/cybba/cybba_segment_automation/Data/Input/Raw_segments/taxonomy_training_prepared.csv.gz"
DEFAULT_STATE_PATH = "/Users/yadnesh/cybba/cybba_segment_automation/Data/Input/Raw_segments/collector_state.json"

# ------------------------------------------------------------
# Shared helpers (mirrors your prepare script style)
# ------------------------------------------------------------


def clean(s: Any) -> str:
    return str(s or "").strip()


def strip_leading_asterisks(s: str) -> str:
    return re.sub(r"^\*+\s*", "", clean(s))


def collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", clean(s)).strip()


def normalize_node(s: str) -> str:
    t = strip_leading_asterisks(s)
    t = collapse_spaces(t)
    t = t.replace("–", "-").replace("—", "-")
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


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_json_maybe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", errors="ignore")
    if isinstance(v, str):
        t = v.strip()
        if not t:
            return None
        try:
            return json.loads(t)
        except Exception:
            return None
    return None


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"last_created_ts": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"last_created_ts": 0}


def save_state(path: Path, last_created_ts: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"last_created_ts": int(last_created_ts)}), encoding="utf-8")

REQUIRED_COLS = ["text", "L1", "L2"]          # Leaf optional but recommended
DEDUP_KEYS = ["text", "L1", "L2", "Leaf"]     # if Leaf missing, we’ll handle it

def append_dedupe_write_gz(out_path: Path, new_df: pd.DataFrame) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize columns
    for c in REQUIRED_COLS:
        if c not in new_df.columns:
            raise ValueError(f"new_df missing required column: {c}")

    # Ensure Leaf exists (even if blank) so dedupe is stable
    if "Leaf" not in new_df.columns:
        new_df = new_df.copy()
        new_df["Leaf"] = ""

    # Read existing (if present)
    if out_path.exists():
        old_df = pd.read_csv(out_path, dtype=str, keep_default_na=False, compression="gzip")
        if "Leaf" not in old_df.columns:
            old_df["Leaf"] = ""
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    # Clean whitespace
    for c in ["text", "L1", "L2", "Leaf"]:
        if c in combined.columns:
            combined[c] = combined[c].astype(str).str.strip()

    # Drop empty labels (keeps training healthy)
    combined = combined[(combined["L1"] != "") & (combined["L2"] != "")]
    # If you require Leaf training:
    # combined = combined[combined["Leaf"] != ""]

    # Dedupe (prefer most recent row by keeping last)
    combined = combined.drop_duplicates(subset=DEDUP_KEYS, keep="last")

    # Atomic write (temp + replace)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{int(time.time())}")
    combined.to_csv(tmp, index=False, encoding="utf-8", compression="gzip")
    os.replace(tmp, out_path)

    return len(combined)

# ------------------------------------------------------------
# Extract L1/L2/Leaf from a segment name like:
#   Cybba > L1 > L2 > Leaf
# ------------------------------------------------------------
def extract_l1_l2_leaf(seg_name: str, provider: str) -> Tuple[str, str, str]:
    parts = split_parts(seg_name)
    if not parts:
        return "", "", ""

    # Drop provider token if present
    if parts and _lower(parts[0]) == _lower(provider):
        parts = parts[1:]

    # Need at least L1, L2, Leaf
    if len(parts) < 3:
        return "", "", ""

    l1 = normalize_node(parts[0])
    l2 = normalize_node(parts[1])
    leaf = normalize_node(parts[-1])
    return l1, l2, leaf


def build_text(row: Dict[str, Any], seg_name: str) -> str:
    # Match your prepared training logic: "name | desc | field | value"
    desc = clean(row.get("Segment Description") or row.get("description") or "")
    field = clean(row.get("Field Name") or row.get("LiveRamp Field Name") or row.get("field_name") or "")
    value = clean(row.get("Value Name") or row.get("LiveRamp Value Name") or row.get("value_name") or "")

    parts = [p for p in [seg_name, desc, field, value] if p]
    return " | ".join(parts)


# ------------------------------------------------------------
# SQLite helpers (runs.db schema tolerant)
# ------------------------------------------------------------
def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [r["name"] for r in cur.fetchall()]


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r["name"] for r in cur.fetchall()]


def pick_first(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def find_runs_table(conn: sqlite3.Connection) -> str:
    # Prefer exact "runs"
    if "runs" in list_tables(conn):
        return "runs"

    # Fallback: any table with run_id + rows
    for t in list_tables(conn):
        cols = table_columns(conn, t)
        if "run_id" in cols and any("rows" in c for c in cols):
            return t

    raise RuntimeError("Could not locate a runs table in runs.db")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to backend/config/config.yml")
    ap.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to backend/data/runs.db")
    ap.add_argument("--out", default=DEFAULT_OUT_PATH, help="Rolling append file (.csv.gz)")
    ap.add_argument("--state", default=DEFAULT_STATE_PATH, help="State file storing last processed created_ts")
    ap.add_argument("--max-runs", type=int, default=300, help="Safety cap: max new runs processed per invocation")
    ap.add_argument(
        "--prefer-final-rows",
        action="store_true",
        help="Prefer final_rows/priced rows from DB if present (recommended).",
    )
    ap.add_argument(
        "--no-prefer-final-rows",
        dest="prefer_final_rows",
        action="store_false",
        help="Use validated rows instead of final_rows if both exist.",
    )
    ap.set_defaults(prefer_final_rows=True)

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    provider = normalize_node(cfg.get("provider", {}).get("cybba_name", "Cybba")) or "Cybba"

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"runs.db not found: {db_path}")

    out_path = Path(args.out).resolve()
    state_path = Path(args.state).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)
    last_created_ts = int(state.get("last_created_ts") or 0)

    logging.info("Provider=%s", provider)
    logging.info("DB=%s", db_path)
    logging.info("OUT=%s", out_path)
    logging.info("STATE=%s (last_created_ts=%d)", state_path, last_created_ts)

    conn = connect_db(db_path)
    runs_table = find_runs_table(conn)
    cols = table_columns(conn, runs_table)

    run_id_col = pick_first(cols, ["run_id", "id"])
    created_col = pick_first(cols, ["created_at", "created_ts", "created", "ts", "timestamp"])
    rows_col = pick_first(cols, ["rows", "rows_json", "validated_rows", "validated"])
    final_rows_col = pick_first(cols, ["final_rows", "final_rows_json", "priced_rows", "final"])

    if not run_id_col:
        raise RuntimeError(f"Table '{runs_table}' missing run_id column. cols={cols}")
    if not created_col:
        raise RuntimeError(f"Table '{runs_table}' missing created timestamp column. cols={cols}")
    if not rows_col and not final_rows_col:
        raise RuntimeError(f"Table '{runs_table}' missing rows/final_rows column. cols={cols}")

    logging.info(
        "Using runs_table=%s run_id_col=%s created_col=%s rows_col=%s final_rows_col=%s",
        runs_table, run_id_col, created_col, rows_col, final_rows_col
    )

    # ----------------------------
    # Load existing append (for dedupe)
    # ----------------------------
    existing_keys: set[str] = set()
    if out_path.exists():
        try:
            old = pd.read_csv(out_path, dtype=str, keep_default_na=False, compression="gzip")
            if "full_path" in old.columns:
                existing_keys = set(old["full_path"].astype(str).str.lower().str.strip().tolist())
            else:
                # fallback dedupe key
                for _, r in old.iterrows():
                    k = "|".join([clean(r.get("L1")), clean(r.get("L2")), clean(r.get("Leaf")), clean(r.get("text"))]).lower()
                    existing_keys.add(k)
            logging.info("Loaded existing append rows=%d keys=%d", len(old), len(existing_keys))
        except Exception as e:
            logging.warning("Could not read existing append file. Will recreate. err=%s", e)

    # ----------------------------
    # Query new runs since last_created_ts
    # ----------------------------
    cur = conn.execute(
        f"""
        SELECT * FROM {runs_table}
        WHERE CAST({created_col} AS INTEGER) > ?
        ORDER BY CAST({created_col} AS INTEGER) ASC
        LIMIT ?
        """,
        (last_created_ts, args.max_runs),
    )
    new_runs = cur.fetchall()

    if not new_runs:
        logging.info("No new runs since last_created_ts=%d. Nothing to do.", last_created_ts)
        return

    logging.info("Found %d new runs.", len(new_runs))

    out_rows: List[Dict[str, Any]] = []
    local_keys: set[str] = set()
    max_seen_ts = last_created_ts

    for rr in new_runs:
        created_ts = int(rr[created_col])
        if created_ts > max_seen_ts:
            max_seen_ts = created_ts

        # Choose payload
        payload = None

        if args.prefer_final_rows and final_rows_col and rr[final_rows_col] is not None:
            payload = read_json_maybe(rr[final_rows_col])

        if payload is None and rows_col and rr[rows_col] is not None:
            payload = read_json_maybe(rr[rows_col])

        if not isinstance(payload, list):
            continue

        for row in payload:
            if not isinstance(row, dict):
                continue

            seg_name = clean(
                row.get("New Segment Name")
                or row.get("Proposed New Segment Name")
                or row.get("Segment Name")
                or ""
            )
            if not seg_name:
                continue

            l1, l2, leaf = extract_l1_l2_leaf(seg_name, provider=provider)
            if not (l1 and l2 and leaf):
                continue

            full_path = f"{provider} > {l1} > {l2} > {leaf}"
            key = full_path.lower().strip()

            if key in existing_keys or key in local_keys:
                continue
            local_keys.add(key)

            text = build_text(row, seg_name)

            out_rows.append(
                {
                    "text": text,
                    "L1": l1,
                    "L2": l2,
                    "Leaf": leaf,
                    "full_path": full_path,
                    "Provider Name": provider,
                    "Segment Name": seg_name,
                }
            )

    # Save state even if nothing new found
    save_state(state_path, max_seen_ts)

    if not out_rows:
        logging.info("No new unique training rows found. Updated state last_created_ts=%d", max_seen_ts)
        return

    add_df = pd.DataFrame(out_rows)

    # ✅ Append + dedupe + atomic rewrite (safe for .gz)
    total_after = append_dedupe_write_gz(out_path, add_df)

    logging.info("Appended %d new rows. Total=%d. last_created_ts=%d", len(add_df), total_after, max_seen_ts)
    print(f"✅ Appended {len(add_df)} rows -> {out_path}")
    print(f"✅ Total rows now {total_after}")
    print(f"✅ Updated state -> {state_path} (last_created_ts={max_seen_ts})")


if __name__ == "__main__":
    main()