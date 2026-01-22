# backend/api.py
from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

# -------------------------------------------------
# Make backend/src importable FIRST
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# now imports from backend/src work reliably
from pipeline import run_pipeline, df_to_csv_bytes, run_pipeline_stream  # noqa: E402


app = FastAPI(title="Cybba Segment Expansion API", version="1.0")

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


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.post("/api/run")
def run() -> Dict[str, Any]:
    """
    Non-streaming run. Returns full rows at once.
    """
    _cleanup_runs()

    try:
        result = run_pipeline(BASE_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    run_id = str(uuid.uuid4())

    validated_df = _sort_validated_for_ui(result["validated_df"].copy())

    RUNS[run_id] = {
        "_created_ts": time.time(),
        "summary": result.get("summary", {}),
        "validated_df": validated_df,
        "final_df": result.get("final_df", pd.DataFrame()),
        "coverage_df": result.get("coverage_df", pd.DataFrame()),
        "proposals_df": result.get("proposals_df", pd.DataFrame()),
    }

    return {
        "run_id": run_id,
        "summary": RUNS[run_id]["summary"],
        "rows": _df_to_rows(validated_df),
    }


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    _cleanup_runs()

    payload = RUNS.get(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="run_id not found (expired or invalid)")

    validated_df = payload["validated_df"]
    return {
        "run_id": run_id,
        "summary": payload.get("summary", {}),
        "rows": _df_to_rows(validated_df),
    }


@app.get("/api/download.csv")
def download_csv(run_id: str, mode: str = "final"):
    _cleanup_runs()

    payload = RUNS.get(run_id)
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
def run_stream():
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

    def sse(event: str, data: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def generator():
        rows_buffer: List[Dict[str, Any]] = []
        final_rows_buffer: List[Dict[str, Any]] = []
        latest_summary: Dict[str, Any] = {}

        # Optional: keep these if pipeline stream ever sends them
        coverage_rows: List[Dict[str, Any]] = []
        proposals_rows: List[Dict[str, Any]] = []

        try:
            # 1) announce run_id immediately so UI can enable downloads early
            yield sse("run_id", {"run_id": run_id})

            # 2) stream pipeline events
            for msg in run_pipeline_stream(BASE_DIR):
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
                    # optional if you add this later
                    if isinstance(data, dict):
                        coverage_rows.append(data)
                    yield sse("coverage_row", data)

                elif ev == "proposal_row":
                    # optional if you add this later
                    if isinstance(data, dict):
                        proposals_rows.append(data)
                    yield sse("proposal_row", data)

                elif ev == "done":
                    # Expecting: {"summary": ..., "rows": [...], "final_rows": [...]}
                    if isinstance(data, dict):
                        latest_summary = data.get("summary", latest_summary) or latest_summary

                        # If pipeline sends complete lists, prefer them:
                        if isinstance(data.get("rows"), list) and data["rows"]:
                            rows_buffer = data["rows"]
                        if isinstance(data.get("final_rows"), list) and data["final_rows"]:
                            final_rows_buffer = data["final_rows"]

                        # Optional: accept these if you later include them
                        if isinstance(data.get("coverage_rows"), list) and data["coverage_rows"]:
                            coverage_rows = data["coverage_rows"]
                        if isinstance(data.get("proposals_rows"), list) and data["proposals_rows"]:
                            proposals_rows = data["proposals_rows"]

                    # 3) build cached dataframes
                    validated_df = pd.DataFrame(rows_buffer)
                    validated_df = _sort_validated_for_ui(validated_df)

                    # If final_rows weren't produced, fall back to validated rows
                    final_df = pd.DataFrame(final_rows_buffer) if final_rows_buffer else validated_df.copy()

                    coverage_df = pd.DataFrame(coverage_rows) if coverage_rows else pd.DataFrame()
                    proposals_df = pd.DataFrame(proposals_rows) if proposals_rows else pd.DataFrame()

                    # 4) cache for download endpoint
                    RUNS[run_id] = {
                        "_created_ts": time.time(),
                        "summary": latest_summary,
                        "validated_df": validated_df,
                        "final_df": final_df,
                        "coverage_df": coverage_df,
                        "proposals_df": proposals_df,
                    }

                    # 5) tell UI we're done (run_id included)
                    yield sse("done", {"run_id": run_id, "summary": latest_summary})
                    return

                else:
                    # passthrough unknown events
                    yield sse(ev or "message", data)

        except Exception as e:
            yield sse("error", {"message": str(e), "run_id": run_id})

    # IMPORTANT: disable buffering/caching for SSE
    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # helps when behind nginx
        },
    )
