import React, { useMemo, useRef, useState } from "react";
import { downloadCsv, runPipelineStream } from "./api";
import RunButton from "../components/RunButton";
import DownloadButton from "../components/DownloadButton";
import SegmentsTable from "../components/SegmentsTable";
import "../styles.css";

type Mode = "final" | "validated";

export default function App() {
  const [streaming, setStreaming] = useState(false);

  // discovered from backend summary
  const [availableValidated, setAvailableValidated] = useState<number | null>(
    null
  );

  // user-controlled cap
  const [maxRows, setMaxRows] = useState<number>(100);

  const [runId, setRunId] = useState<string | null>(null);
  const [summary, setSummary] = useState<Record<string, any> | null>(null);

  // validated rows (streaming)
  const [rows, setRows] = useState<Record<string, any>[]>([]);

  // FINAL rows (priced, template-driven)
  const [finalRows, setFinalRows] = useState<Record<string, any>[]>([]);

  const [error, setError] = useState<string | null>(null);

  const stopRef = useRef<null | (() => void)>(null);

  // filters
  const [query, setQuery] = useState("");
  const [minUniq, setMinUniq] = useState<number>(0);

  // Once pricing is done, always show finalRows. While streaming, show validated rows.
  const displayRows = finalRows.length > 0 ? finalRows : rows;

  const filteredRows = useMemo(() => {
    const q = query.trim().toLowerCase();

    return displayRows.filter((r) => {
      const name = String(
        r["New Segment Name"] ?? r["Proposed New Segment Name"] ?? ""
      ).toLowerCase();

      const okQ = !q || name.includes(q);

      // uniqueness might not exist in final rows; if missing, don't filter it out
      const uniqRaw = r["uniqueness_score"];
      const uniq = Number(uniqRaw);
      const okUniq =
        uniqRaw === undefined || uniqRaw === null || uniqRaw === ""
          ? true
          : Number.isFinite(uniq)
          ? uniq >= minUniq
          : true;

      return okQ && okUniq;
    });
  }, [displayRows, query, minUniq]);

  function onRun() {
  try {
    if (stopRef.current) stopRef.current();

    setStreaming(true);
    setError(null);

    setRows([]);
    setFinalRows([]);
    setSummary(null);
    setRunId(null);

    const stop = runPipelineStream({
      max_rows: maxRows,
      onRunId: setRunId,
      onSummary: (s) => {
        setSummary(s ?? null);
        const total = Number((s as any)?.validated_total);
        if (Number.isFinite(total) && total > 0) {
          setAvailableValidated(total);
          setMaxRows((prev) => Math.min(prev, total));
        }
      },
      onRow: (row) => setRows((prev) => [...prev, row]),
      onDone: (payload) => {
        if (payload?.summary) setSummary(payload.summary);
        if (payload?.run_id) setRunId(payload.run_id);
        if (Array.isArray(payload?.final_rows)) setFinalRows(payload.final_rows);
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
  } catch (e: any) {
    setError(e?.message || String(e));
    setStreaming(false);
    stopRef.current = null;
  }
}


  function onDownload(mode: Mode) {
    if (!runId) return;
    downloadCsv(runId, mode);
  }

  const showOverlay = streaming && rows.length === 0;

  const sliderMax = availableValidated ?? 500;
  const sliderValue = Math.max(1, Math.min(maxRows, sliderMax));

  return (
    <div className="app">
      <div className="bg" />

      <header className="header">
        <div className="titleBlock">
          <div className="badge">Cybba</div>
          <h1 className="title">Segment Expansion</h1>
          <p className="subtitle">Generate, price, and export new audience segments.</p>
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
        {/* error toast */}
        <div className={`toast ${error ? "toastShow" : ""}`}>
          <div className="toastTitle">Backend error</div>
          <div className="toastBody">{error ?? ""}</div>
        </div>

        {/* filters */}
        <section className="filtersStrip fadeInUp">
          <div className="filtersStripHeader">
            <div className="cardTitle">Filters</div>
            <div className="cardHint">
              Showing <b>{filteredRows.length}</b> / {displayRows.length}
            </div>
          </div>

          <div className="filtersStripBody">
            <div className="field">
              <span>
                Segments to generate{" "}
                {availableValidated ? (
                  <span className="pill" style={{ marginLeft: 8 }}>
                    {availableValidated} available
                  </span>
                ) : null}
              </span>

              <div className="sliderRow">
                <input
                  type="range"
                  className={`slider ${streaming ? "sliderDisabled" : ""}`}
                  min={1}
                  max={sliderMax}
                  value={sliderValue}
                  disabled={streaming}
                  onChange={(e) => setMaxRows(Number(e.target.value))}
                />
                <div className={`sliderValue ${streaming ? "sliderValueDisabled" : ""}`}>
                  {sliderValue}
                </div>
              </div>
            </div>

            <label className="field">
              <span>Search</span>
              <input value={query} onChange={(e) => setQuery(e.target.value)} />
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
        </section>

        {/* table */}
        <section className="card fadeInUp">
          <div className="cardHeader">
            <div className="cardTitle">
              {finalRows.length > 0 ? "Priced Segments" : "Validated Segments"}
            </div>

            {streaming ? (
              <div className="muted" style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
                <span className="miniSpinner" />
                streaming…
              </div>
            ) : null}
          </div>

          <SegmentsTable rows={filteredRows} loading={streaming && rows.length === 0} />
        </section>
      </main>

      {/* overlay */}
      <div className={`overlay ${showOverlay ? "overlayShow" : ""}`}>
        <div className="overlayCard">
          <div className="spinner" />
          <div className="overlayText">
            <div className="overlayTitle">Running pipeline</div>
            <div className="overlaySub">Generating, pricing, and ranking segments…</div>
          </div>
        </div>
      </div>
    </div>
  );
}
