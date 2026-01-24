import React, { useMemo } from "react";

function fmt3(x: any) {
  return typeof x === "number" && Number.isFinite(x) ? x.toFixed(3) : "—";
}

function fmtNice(x: any) {
  if (typeof x === "number" && Number.isFinite(x)) {
    // show whole numbers without decimals, else 2 decimals
    return Number.isInteger(x) ? String(x) : x.toFixed(2);
  }
  // try numeric strings
  const n = Number(x);
  if (Number.isFinite(n)) return Number.isInteger(n) ? String(n) : n.toFixed(2);
  return x == null || x === "" ? "—" : String(x);
}

function StatsBlock({ obj }: { obj: any }) {
  if (!obj || typeof obj !== "object") return <span className="muted">—</span>;
  return (
    <div style={{ display: "grid", gap: 4 }}>
      <div className="muted">Min: {fmt3(obj.min)}</div>
      <div className="muted">P10: {fmt3(obj.p10)}</div>
      <div className="muted">P50: {fmt3(obj.p50)}</div>
      <div className="muted">P90: {fmt3(obj.p90)}</div>
      <div className="muted">Mean: {fmt3(obj.mean)}</div>
    </div>
  );
}

function MetricTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
}) {
  return (
    <div className="metricTile">
      <div className="metricLabel">{label}</div>
      <div className="metricValue">{value}</div>
      {hint ? <div className="metricHint">{hint}</div> : null}
    </div>
  );
}

export default function SummaryCard({
  runId,
  summary,
  loading,
  variant = "full",
  streaming = false,
}: {
  runId: string | null;
  summary: Record<string, any> | null;
  loading: boolean;
  variant?: "full" | "compact";
  streaming?: boolean;
}) {
  const renderVal = (k: string, v: any) => {
    if (k === "uniqueness_stats" || k === "rank_stats") {
      return <StatsBlock obj={v} />;
    }
    return String(v);
  };

  // pick the key metrics for the compact strip
  const tiles = useMemo(() => {
    const s = summary ?? {};
    const totalProposals = s.total_proposals;
    const validatedTotal = s.validated_total ?? s.validated; // fallback
    const validatedGenerated = s.validated_generated ?? null;
    const covered = s.covered;

    const dupes = s.duplicates_within_output_normalized;
    const phase = s.phase;

    return [
      {
        label: "Run",
        value: runId ? <code>{runId}</code> : "No run yet",
        hint: streaming ? (
          <span style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
            <span className="miniSpinner" />
            streaming…
          </span>
        ) : phase ? (
          <span className="muted">phase: {String(phase)}</span>
        ) : undefined,
      },
      {
        label: "Total proposals",
        value: fmtNice(totalProposals),
      },
      {
        label: "Validated available",
        value: fmtNice(validatedTotal),
      },
      {
        label: "Generated now",
        value: validatedGenerated != null ? fmtNice(validatedGenerated) : "—",
        hint: <span className="muted">slider controls this</span>,
      },
      {
        label: "Covered",
        value: fmtNice(covered),
      },
      {
        label: "Dupes (normalized)",
        value: fmtNice(dupes),
      },
    ];
  }, [summary, runId, streaming]);

  // ========== COMPACT ==========
  if (variant === "compact") {
    // If no run yet, still show a clean strip (but empty values)
    return (
      <div className="metricsStrip">
        {tiles.map((t) => (
          <MetricTile key={t.label} label={t.label} value={t.value} hint={t.hint} />
        ))}
      </div>
    );
  }

  // ========== FULL (existing behavior) ==========
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
