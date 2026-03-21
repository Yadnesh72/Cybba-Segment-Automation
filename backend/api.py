# backend/api.py
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from src.pipeline import load_config
from src.comparison import router as comparison_router
from src.suggestions import router as suggestions_router

import pandas as pd
from fastapi import Body, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from src.persist import (
    get_last_run,   # ✅ changed
    get_run_payload,
    init_db,
    set_last_run,
    clear_last_run,  # ✅ new
    upsert_run,
)

# -------------------------------------------------
# Make backend/src importable FIRST
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


# now imports from backend/src work reliably
from pipeline import df_to_csv_bytes, run_pipeline, run_pipeline_stream  # noqa: E402
from src.analytics import router as analytics_router


# -------------------------------------------------
# Persistence (SQLite)
# -------------------------------------------------
from persist import (  # noqa: E402
    get_last_run_id,
    get_run_payload,
    init_db,
    set_last_run,
    upsert_run,
)

DB_PATH = BASE_DIR / "data" / "runs.db"
init_db(DB_PATH)

app = FastAPI(title="Cybba Segment Expansion API", version="1.0")
app.include_router(analytics_router)
app.include_router(comparison_router)
app.include_router(suggestions_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite
        "http://localhost:3000",  # CRA/Next dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# In-memory run cache (dev-friendly)
# -------------------------------------------------
RUNS: Dict[str, Dict[str, Any]] = {}
RUN_TTL_SECONDS = 60 * 30  # 30 minutes


def _cleanup_runs() -> None:
    now = time.time()
    dead: List[str] = []
    for run_id, payload in RUNS.items():
        created = float(payload.get("_created_ts", 0))
        if now - created > RUN_TTL_SECONDS:
            dead.append(run_id)
    for rid in dead:
        RUNS.pop(rid, None)


def _df_to_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    safe = df.copy()
    safe = safe.replace({pd.NA: None})
    safe = safe.where(pd.notnull(safe), None)
    return safe.to_dict(orient="records")


def _sort_validated_for_ui(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "rank_score" in out.columns:
        return out.sort_values("rank_score", ascending=False)
    if "Composition Similarity" in out.columns:
        return out.sort_values("Composition Similarity", ascending=False)
    return out


def _resolve_user_id(
    x_user_id: Optional[str],
    user_id: Optional[str],
) -> str:
    # minimal + backward compatible: if not provided, fall back to "anonymous"
    uid = (x_user_id or user_id or "").strip()
    return uid or "anonymous"


def _load_run_into_memory_from_db(run_id: str) -> Optional[Dict[str, Any]]:
    payload = get_run_payload(DB_PATH, run_id=run_id)
    if not payload:
        return None

    # Rebuild dataframes for download endpoint compatibility
    validated_rows = payload.get("rows") or []
    final_rows = payload.get("final_rows") or []

    validated_df = pd.DataFrame(validated_rows)
    validated_df = _sort_validated_for_ui(validated_df)

    final_df = pd.DataFrame(final_rows) if final_rows else validated_df.copy()

    mem = {
        "_created_ts": time.time(),
        "summary": payload.get("summary", {}),
        "validated_df": validated_df,
        "final_df": final_df,
        "coverage_df": pd.DataFrame(),  # not persisted (yet)
        "proposals_df": pd.DataFrame(),  # not persisted (yet)
    }
    RUNS[run_id] = mem
    return mem


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True}

@app.get("/api/catalog/cybba")
def get_cybba_catalog(
    q: Optional[str] = Query(default=None, description="Optional search term"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
) -> Dict[str, Any]:
    """
    Returns rows from Input A (full catalog) filtered to Provider=cybba.
    Supports optional search and paging.
    """
    cfg = load_config(BASE_DIR)

    input_dir = Path(cfg["paths"]["input_dir"])
    input_a = input_dir / cfg["files"]["input_a"]
    if not input_a.exists():
        raise HTTPException(status_code=404, detail=f"Input A not found: {input_a}")

    df = pd.read_csv(input_a, dtype=str, keep_default_na=False)

    provider_col = "Provider Name"
    if provider_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Input A missing column: {provider_col}")

    cybba_name = str(cfg.get("provider", {}).get("cybba_name", "Cybba")).strip().lower()
    out = df[df[provider_col].astype(str).str.strip().str.lower() == cybba_name].copy()

    # optional search across Segment Name / Description
    if q:
        qq = str(q).strip().lower()
        name_col = "Segment Name"
        desc_col = "Segment Description"
        cols = [c for c in [name_col, desc_col] if c in out.columns]
        if cols:
            mask = False
            for c in cols:
                mask = mask | out[c].astype(str).str.lower().str.contains(qq, na=False)
            out = out[mask]

    total = int(len(out))
    start = (page - 1) * page_size
    end = start + page_size
    page_df = out.iloc[start:end].copy()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "rows": _df_to_rows(page_df),
        "columns": list(page_df.columns),
    }

# ----------------------------
# Per-user: last run endpoints
# ----------------------------
@app.get("/api/users/me/last-run")
def api_get_last_run(
    user_id: Optional[str] = Query(default=None),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
) -> Dict[str, Any]:
    uid = _resolve_user_id(x_user_id, user_id)
    rec = get_last_run(DB_PATH, user_id=uid)
    return rec or {"run_id": None, "updated_at": None}


@app.post("/api/users/me/last-run")
def api_set_last_run(
    body: Dict[str, Any],
    user_id: Optional[str] = Query(default=None),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
) -> Dict[str, Any]:
    uid = _resolve_user_id(x_user_id, user_id)
    rid = str((body or {}).get("run_id") or "").strip()
    if not rid:
        raise HTTPException(status_code=400, detail="Missing run_id")
    set_last_run(DB_PATH, user_id=uid, run_id=rid)
    rec = get_last_run(DB_PATH, user_id=uid)
    return rec or {"run_id": rid, "updated_at": None}

@app.delete("/api/users/me/last-run")
def api_clear_last_run(
    user_id: Optional[str] = Query(default=None),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
) -> Dict[str, Any]:
    uid = _resolve_user_id(x_user_id, user_id)
    clear_last_run(DB_PATH, user_id=uid)
    return {"ok": True}

@app.post("/api/run")
def run(
    max_rows: Optional[int] = Query(
        default=None,
        ge=1,
        description="Cap the number of validated rows to generate (post-validation).",
    ),
    user_id: Optional[str] = Query(default=None),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
) -> Dict[str, Any]:
    """
    Non-streaming run. Returns full rows at once.
    """
    _cleanup_runs()

    uid = _resolve_user_id(x_user_id, user_id)

    try:
        result = run_pipeline(BASE_DIR, max_rows=max_rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    run_id = str(uuid.uuid4())

    validated_df = _sort_validated_for_ui(result["validated_df"].copy())
    final_df = result.get("final_df", pd.DataFrame())

    final_df = final_df if isinstance(final_df, pd.DataFrame) else pd.DataFrame(final_df)

    RUNS[run_id] = {
        "_created_ts": time.time(),
        "summary": result.get("summary", {}),
        "validated_df": validated_df,
        "final_df": final_df,
        "coverage_df": result.get("coverage_df", pd.DataFrame()),
        "proposals_df": result.get("proposals_df", pd.DataFrame()),
    }

    # Persist (rows + final_rows + summary) and set last run for this user
    try:
        upsert_run(
            DB_PATH,
            run_id=run_id,
            created_by=uid,
            summary=RUNS[run_id]["summary"] or {},
            rows=_df_to_rows(validated_df),
            final_rows=_df_to_rows(final_df) if not final_df.empty else [],
        )
        set_last_run(DB_PATH, user_id=uid, run_id=run_id)
    except Exception:
        # Persistence failures should not break the run in dev mode
        pass

    return {
        "run_id": run_id,
        "summary": RUNS[run_id]["summary"],
        "rows": _df_to_rows(validated_df),
        # include final_rows for pricing UI
        "final_rows": _df_to_rows(final_df) if not final_df.empty else [],
    }


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    _cleanup_runs()

    payload = RUNS.get(run_id)
    if not payload:
        payload = _load_run_into_memory_from_db(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="run_id not found (expired or invalid)")

    validated_df = payload["validated_df"]
    final_df = payload.get("final_df", pd.DataFrame())

    return {
        "run_id": run_id,
        "summary": payload.get("summary", {}),
        "rows": _df_to_rows(validated_df),
        "final_rows": _df_to_rows(final_df) if isinstance(final_df, pd.DataFrame) and not final_df.empty else [],
    }


@app.get("/api/download.csv")
def download_csv(run_id: str, mode: str = "final"):
    _cleanup_runs()

    payload = RUNS.get(run_id)
    if not payload:
        payload = _load_run_into_memory_from_db(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="run_id not found (expired or invalid)")

    today = date.today().isoformat()  # YYYY-MM-DD

    if mode == "final":
        df = payload["final_df"]
        filename = f"Cybba_New_Additional_Segments_{today}.csv"
    elif mode == "validated":
        df = payload["validated_df"]
        filename = f"proposals_validated_{today}.csv"
    elif mode == "proposals":
        df = payload["proposals_df"]
        filename = f"proposals_{today}.csv"
    elif mode == "coverage":
        df = payload["coverage_df"]
        filename = f"coverage_{today}.csv"
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use final|validated|proposals|coverage")

    csv_bytes = df_to_csv_bytes(df)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/run/stream")
def run_stream(
    max_rows: Optional[int] = Query(
        default=None,
        ge=1,
        description="Cap the number of validated rows to stream/generate (post-validation).",
    ),
    user_id: Optional[str] = Query(
        default=None,
        description="User identifier for per-user persistence (EventSource cannot send headers, so pass query param).",
    ),
    enable_descriptions: Optional[bool] = Query(default=None),
    enable_pricing: Optional[bool] = Query(default=None),
    enable_taxonomy: Optional[bool] = Query(default=None),
    enable_coverage: Optional[bool] = Query(default=None),
    enable_llm_generation: Optional[bool] = Query(default=None),
    enable_llm_web_assistance: Optional[bool] = Query(default=None),
):
    """
    SSE stream. Use EventSource on the frontend (GET-only).
    Streams:
      event: run_id  -> run_id immediately (so downloads can be enabled)
      event: summary -> early summary
      event: row     -> one row at a time
      event: done    -> final summary (+ run_id)
      event: error   -> error payload

    Also caches the run at the end so /api/download.csv works.
    """
    _cleanup_runs()
    run_id = str(uuid.uuid4())

    uid = _resolve_user_id(None, user_id)

    def sse(event: str, data: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def generator():
        rows_buffer: List[Dict[str, Any]] = []
        final_rows_buffer: List[Dict[str, Any]] = []
        latest_summary: Dict[str, Any] = {}

        coverage_rows: List[Dict[str, Any]] = []
        proposals_rows: List[Dict[str, Any]] = []

        try:
            # 1) announce run_id immediately so UI can enable downloads early
            yield sse("run_id", {"run_id": run_id})
            RUNS[run_id] = {
                "_created_ts": time.time(),
                "summary": {},
                "validated_df": pd.DataFrame(),
                "final_df": pd.DataFrame(),
                "coverage_df": pd.DataFrame(),
                "proposals_df": pd.DataFrame(),
            }

            try:
                upsert_run(DB_PATH, run_id=run_id, created_by=uid, summary={}, rows=[], final_rows=[])
                set_last_run(DB_PATH, user_id=uid, run_id=run_id)
            except Exception:
                pass

            # 2) stream pipeline events
            overrides = {
            "enable_descriptions": enable_descriptions,
            "enable_pricing": enable_pricing,
            "enable_taxonomy": enable_taxonomy,
            "enable_coverage": enable_coverage,
            "enable_llm_generation": enable_llm_generation,
            "enable_llm_web_assistance": enable_llm_web_assistance,
        }

            for msg in run_pipeline_stream(BASE_DIR, max_rows=max_rows, overrides=overrides):
                ev = msg.get("event")
                data = msg.get("data", {})

                if ev == "summary":
                    latest_summary = data if isinstance(data, dict) else {}
                    yield sse("summary", latest_summary)

                elif ev == "row":
                    if isinstance(data, dict):
                        rows_buffer.append(data)
                    yield sse("row", data)

                elif ev == "coverage_row":
                    if isinstance(data, dict):
                        coverage_rows.append(data)
                    yield sse("coverage_row", data)

                elif ev == "proposal_row":
                    if isinstance(data, dict):
                        proposals_rows.append(data)
                    yield sse("proposal_row", data)

                elif ev == "done":
                    # Expecting: {"summary": ..., "rows": [...], "final_rows": [...]}
                    if isinstance(data, dict):
                        latest_summary = data.get("summary", latest_summary) or latest_summary
                        # Prefer the longer list (prevents "done" payload from truncating streamed buffers)
                        done_rows = data.get("rows")
                        if isinstance(done_rows, list):
                            if (not rows_buffer) or (len(done_rows) > len(rows_buffer)):
                                rows_buffer = done_rows

                        done_final = data.get("final_rows")
                        if isinstance(done_final, list):
                            if (not final_rows_buffer) or (len(done_final) > len(final_rows_buffer)):
                                final_rows_buffer = done_final

                        done_cov = data.get("coverage_rows")
                        if isinstance(done_cov, list):
                            if (not coverage_rows) or (len(done_cov) > len(coverage_rows)):
                                coverage_rows = done_cov

                        done_prop = data.get("proposals_rows")
                        if isinstance(done_prop, list):
                            if (not proposals_rows) or (len(done_prop) > len(proposals_rows)):
                                proposals_rows = done_prop

                    # build cached dataframes
                    validated_df = pd.DataFrame(rows_buffer)
                    validated_df = _sort_validated_for_ui(validated_df)

                    final_df = pd.DataFrame(final_rows_buffer) if final_rows_buffer else validated_df.copy()
                    coverage_df = pd.DataFrame(coverage_rows) if coverage_rows else pd.DataFrame()
                    proposals_df = pd.DataFrame(proposals_rows) if proposals_rows else pd.DataFrame()

                    RUNS[run_id] = {
                        "_created_ts": time.time(),
                        "summary": latest_summary,
                        "validated_df": validated_df,
                        "final_df": final_df,
                        "coverage_df": coverage_df,
                        "proposals_df": proposals_df,
                    }

                    # Persist + set per-user last run
                    try:
                        upsert_run(
                            DB_PATH,
                            run_id=run_id,
                            created_by=uid,
                            summary=latest_summary or {},
                            rows=rows_buffer or [],
                            final_rows=final_rows_buffer or [],
                        )
                        set_last_run(DB_PATH, user_id=uid, run_id=run_id)
                    except Exception:
                        pass

                    yield sse(
                        "done",
                        {
                            "run_id": run_id,
                            "summary": latest_summary,
                            "rows": rows_buffer,
                            "final_rows": final_rows_buffer,
                        },
                    )
                    return

                else:
                    yield sse(ev or "message", data)

        except Exception as e:
            import traceback
            logging.error("Stream failed: %s\n%s", e, traceback.format_exc())
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )