import React, { useMemo, useRef, useState } from "react";
import { downloadCsv, runPipelineStream } from "./api";
import RunButton from "../components/RunButton";
import DownloadButton from "../components/DownloadButton";
import SummaryCard from "../components/SummaryCard";
import SegmentsTable from "../components/SegmentsTable";
import "../styles.css";

type Mode = "final" | "validated";

export default function App() {
  const [streaming, setStreaming] = useState(false); // ✅ replaces "loading"
  const [runId, setRunId] = useState<string | null>(null);
  const [summary, setSummary] = useState<Record<string, any> | null>(null);
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const [error, setError] = useState<string | null>(null);

  const stopRef = useRef<null | (() => void)>(null);

  // filters
  const [query, setQuery] = useState("");
  const [minUniq, setMinUniq] = useState<number>(0);

  const filteredRows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return rows.filter((r) => {
      const name = String(r["Proposed New Segment Name"] ?? "").toLowerCase();
      const okQ = !q || name.includes(q);
      const uniq = Number(r["uniqueness_score"]);
      const okUniq = Number.isFinite(uniq) ? uniq >= minUniq : true;
      return okQ && okUniq;
    });
  }, [rows, query, minUniq]);

  function onRun() {
    if (stopRef.current) {
      stopRef.current();
      stopRef.current = null;
    }

    setStreaming(true);
    setError(null);

    setRows([]);
    setSummary(null);
    setRunId(null);

    const stop = runPipelineStream({
      onRunId: (id) => setRunId(id),
      onSummary: (s) => setSummary(s ?? null),

      onRow: (row) => {
        setRows((prev) => [...prev, row]);
        // ✅ overlay will auto-hide because rows.length becomes > 0
      },

      onDone: (payload) => {
        if (payload?.summary) setSummary(payload.summary);
        if (payload?.run_id) setRunId(payload.run_id);

        setStreaming(false);
        stopRef.current = null;
      },

      onError: (msg) => {
        setError(msg);
        setStreaming(false);
        stopRef.current = null;
      },
    });

    stopRef.current = stop;
  }

  function onDownload(mode: Mode) {
    if (!runId) return;
    downloadCsv(runId, mode);
  }

  // ✅ Overlay visible ONLY until first row arrives
  const showOverlay = streaming && rows.length === 0;

  return (
    <div className="app">
      <div className="bg" />

      <header className="header">
        <div className="titleBlock">
          <div className="badge">Cybba</div>
          <h1 className="title">Segment Expansion</h1>
          <p className="subtitle">
            Run the pipeline, review validated segments, and download the CSV.
          </p>
        </div>

        <div className="actions">
          <RunButton onClick={onRun} loading={streaming} />
          <DownloadButton
            disabled={!runId || streaming}
            onDownloadFinal={() => onDownload("final")}
            onDownloadValidated={() => onDownload("validated")}
          />
        </div>
      </header>

      <main className="main">
        <div className={`toast ${error ? "toastShow" : ""}`}>
          <div className="toastTitle">Backend error</div>
          <div className="toastBody">{error ?? ""}</div>
        </div>

        <section className="grid">
          <div className="card fadeInUp">
            <SummaryCard runId={runId} summary={summary} loading={streaming && rows.length === 0} />
          </div>

          <div className="card fadeInUp" style={{ animationDelay: "60ms" }}>
            <div className="cardHeader">
              <div className="cardTitle">Filters</div>
              <div className="cardHint">
                Showing <b>{filteredRows.length}</b> / {rows.length}
              </div>
            </div>

            <div className="filters">
              <label className="field">
                <span>Search name</span>
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder='e.g. "Decision Makers"'
                />
              </label>

              <label className="field">
                <span>Min uniqueness</span>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={minUniq}
                  onChange={(e) => setMinUniq(Number(e.target.value))}
                />
              </label>
            </div>

            <div className="muted" style={{ marginTop: 8 }}>
              Tip: sort by <code>rank_score</code> on the table.
            </div>
          </div>
        </section>

        <section className="card fadeInUp" style={{ animationDelay: "120ms" }}>
          <div className="cardHeader">
            <div className="cardTitle">Validated Proposals</div>

            {/* ✅ Small loading symbol while streaming */}
            <div className="muted" style={{ display: "flex", gap: 10, alignItems: "center" }}>
              {runId ? (
                <>
                  run_id: <code>{runId}</code>
                </>
              ) : (
                <>Run the pipeline to view results.</>
              )}

              {streaming ? (
                <>
                  <span className="miniSpinner" />
                  <span>streaming…</span>
                </>
              ) : null}
            </div>
          </div>

          <SegmentsTable rows={filteredRows} loading={streaming && rows.length === 0} />
        </section>
      </main>

      {/* ✅ Big overlay only until first row */}
      <div className={`overlay ${showOverlay ? "overlayShow" : ""}`}>
        <div className="overlayCard">
          <div className="spinner" />
          <div className="overlayText">
            <div className="overlayTitle">Running pipeline</div>
            <div className="overlaySub">
              Generating, validating, ranking… streaming rows as they’re ready.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
