import React from "react";

function fmt(x: any) {
  return typeof x === "number" && Number.isFinite(x) ? x.toFixed(3) : "—";
}

function StatsBlock({ obj }: { obj: any }) {
  if (!obj || typeof obj !== "object") return <span className="muted">—</span>;
  return (
    <div style={{ display: "grid", gap: 4 }}>
      <div className="muted">Min: {fmt(obj.min)}</div>
      <div className="muted">P10: {fmt(obj.p10)}</div>
      <div className="muted">P50: {fmt(obj.p50)}</div>
      <div className="muted">P90: {fmt(obj.p90)}</div>
      <div className="muted">Mean: {fmt(obj.mean)}</div>
    </div>
  );
}

export default function SummaryCard({
  runId,
  summary,
  loading,
}: {
  runId: string | null;
  summary: Record<string, any> | null;
  loading: boolean;
}) {
  const renderVal = (k: string, v: any) => {
    if (k === "uniqueness_stats" || k === "rank_stats") {
      return <StatsBlock obj={v} />;
    }
    return String(v);
  };

  return (
    <div>
      <div className="cardHeader">
        <div className="cardTitle">Run Summary</div>
        <div className="muted">{runId ? <code>{runId}</code> : "No run yet"}</div>
      </div>

      {!runId ? (
        <div className="empty">
          <div className="emptyTitle">No results</div>
          <div className="muted">Click “Run Pipeline” to generate segments.</div>
        </div>
      ) : loading ? (
        <div className="skeletonGrid">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="skeletonTile">
              <div className="skeletonLine sm" />
              <div className="skeletonLine" />
            </div>
          ))}
        </div>
      ) : (
        <div className="kvGrid">
          {summary
            ? Object.entries(summary).map(([k, v]) => (
                <div key={k} className="kv">
                  <div className="kvKey">{k}</div>
                  <div className="kvVal">{renderVal(k, v)}</div>
                </div>
              ))
            : null}
        </div>
      )}
    </div>
  );
}
