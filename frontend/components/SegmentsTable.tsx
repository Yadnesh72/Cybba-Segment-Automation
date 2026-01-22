import React, { useMemo, useState } from "react";

const NUM_KEYS = new Set([
  "rank_score",
  "uniqueness_score",
  "Composition Similarity",
  "Closest Cybba Similarity",
]);

function toNum(v: any): number | null {
  if (v === null || v === undefined || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

function fmtNum(v: any, digits = 1): string {
  const n = toNum(v);
  if (n === null) return "";
  return n.toFixed(digits);
}

export default function SegmentsTable({
  rows,
  loading,
}: {
  rows: Record<string, any>[];
  loading: boolean;
}) {
  const [sortKey, setSortKey] = useState("rank_score");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [pageSize, setPageSize] = useState(50);

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = a?.[sortKey];
      const bv = b?.[sortKey];

      const an = toNum(av);
      const bn = toNum(bv);

      // numeric sort if both numbers
      if (an !== null && bn !== null) {
        if (an < bn) return sortDir === "asc" ? -1 : 1;
        if (an > bn) return sortDir === "asc" ? 1 : -1;
        return 0;
      }

      // if one numeric and the other not, numeric wins
      if (an !== null && bn === null) return sortDir === "asc" ? -1 : 1;
      if (an === null && bn !== null) return sortDir === "asc" ? 1 : -1;

      // fallback string sort
      const as = String(av ?? "").toLowerCase();
      const bs = String(bv ?? "").toLowerCase();
      if (as < bs) return sortDir === "asc" ? -1 : 1;
      if (as > bs) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  const visible = useMemo(() => sorted.slice(0, pageSize), [sorted, pageSize]);

  const columns = useMemo(() => {
    const base = [
      "Proposed New Segment Name",
      "Segment Description",
      "rank_score",
      "uniqueness_score",
      "Composition Similarity",
      "Closest Cybba Similarity",
    ];
    return base.filter((c) => rows.some((r) => r?.[c] !== undefined));
  }, [rows]);

  // ✅ Only show skeleton when loading AND there are no rows yet
  if (loading && rows.length === 0) {
    return (
      <div className="tableSkeleton">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="skeletonRow">
            <div className="skeletonBar" />
          </div>
        ))}
      </div>
    );
  }

  // we keep it minimal + good looking without forcing you to rewrite styles.css:
  const nameCellStyle: React.CSSProperties = {
    fontWeight: 900,
    lineHeight: 1.25,
    letterSpacing: "-0.2px",
  };

  const descCellStyle: React.CSSProperties = {
    color: "rgba(255,255,255,0.82)",
    lineHeight: 1.35,
  };

  const chipStyle: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    padding: "6px 10px",
    borderRadius: 999,
    border: "1px solid rgba(255,255,255,0.14)",
    background: "rgba(0,0,0,0.14)",
    fontWeight: 800,
    fontSize: 12,
    fontVariantNumeric: "tabular-nums",
  };

  return (
    <>
      <div className="tableToolbar">
        <div className="muted" style={{ display: "inline-flex", gap: 10, alignItems: "center" }}>
          <span>
            Showing {Math.min(pageSize, rows.length)} / {rows.length}
          </span>

          {loading ? (
            <span style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
              <span
                className="btnSpinner btnSpinnerShow"
                style={{ display: "inline-block" }}
              />
              <span>streaming…</span>
            </span>
          ) : null}
        </div>

        <div className="tableControls">
          <label className="miniField">
            Sort
            <select value={sortKey} onChange={(e) => setSortKey(e.target.value)}>
              <option value="rank_score">rank_score</option>
              <option value="uniqueness_score">uniqueness_score</option>
              <option value="Composition Similarity">Composition Similarity</option>
              <option value="Closest Cybba Similarity">Closest Cybba Similarity</option>
            </select>
          </label>

          <label className="miniField">
            Dir
            <select value={sortDir} onChange={(e) => setSortDir(e.target.value as any)}>
              <option value="desc">desc</option>
              <option value="asc">asc</option>
            </select>
          </label>

          <label className="miniField">
            Rows
            <select value={pageSize} onChange={(e) => setPageSize(Number(e.target.value))}>
              {[25, 50, 100, 250, 500].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="tableWrap">
        <table className="table">
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>

          <tbody>
            {visible.map((r, idx) => {
              const key = r?.["Proposed New Segment Name"] ?? `${idx}`;
              return (
                <tr key={key}>
                  {columns.map((c) => {
                    const v = r?.[c];

                    // emphasized cells
                    if (c === "Proposed New Segment Name") {
                      return (
                        <td key={c}>
                          <div style={nameCellStyle}>{String(v ?? "")}</div>
                        </td>
                      );
                    }

                    if (c === "Segment Description") {
                      return (
                        <td key={c}>
                          <div style={descCellStyle}>{String(v ?? "")}</div>
                        </td>
                      );
                    }

                    // numeric chips (rounded)
                    if (NUM_KEYS.has(c)) {
                      const out = fmtNum(v, 1);
                      return (
                        <td key={c}>
                          <span style={chipStyle}>{out}</span>
                        </td>
                      );
                    }

                    // default
                    return <td key={c}>{String(v ?? "")}</td>;
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
