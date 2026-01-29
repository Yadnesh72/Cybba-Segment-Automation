import React, { useEffect, useMemo, useState } from "react";

/** Pricing columns (final output) */
const PRICE_COLS = [
  "Digital Ad Targeting Price (CPM)",
  "Content Marketing Price (CPM)",
  "TV Targeting Price (CPM)",
  "Cost Per Click",
  "Programmatic % of Media",
  "CPM Cap",
  "Advertiser Direct % of Media",
] as const;

const PCT_COLS = new Set<string>([
  "Programmatic % of Media",
  "Advertiser Direct % of Media",
]);

const CPM_COLS = new Set<string>([
  "Digital Ad Targeting Price (CPM)",
  "Content Marketing Price (CPM)",
  "TV Targeting Price (CPM)",
  "CPM Cap",
]);

const CPC_COLS = new Set<string>(["Cost Per Click"]);

const SCORE_COLS = new Set<string>([
  "rank_score",
  "uniqueness_score",
  "Composition Similarity",
  "Closest Cybba Similarity",
]);

/** ✅ UI-only abbreviations (does NOT affect CSV output) */
const COL_LABELS: Record<string, string> = {
  "New Segment Name": "Segment",
  "Segment Description": "Description",

  "Digital Ad Targeting Price (CPM)": "D-CPM",
  "Content Marketing Price (CPM)": "C-CPM",
  "TV Targeting Price (CPM)": "TV-CPM",
  "Cost Per Click": "CPC",
  "Programmatic % of Media": "Prog%",
  "CPM Cap": "Cap",
  "Advertiser Direct % of Media": "Direct%",

  rank_score: "Rank",
  uniqueness_score: "Uniq",
  "Composition Similarity": "Comp",
  "Closest Cybba Similarity": "Closest",
};

function toNum(v: any): number | null {
  if (v === null || v === undefined || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

/** ✅ Prices should show $ */
function fmtMoney(v: any, digits: number): string {
  const n = toNum(v);
  if (n === null) return "";
  return `$${n.toFixed(digits)}`;
}

function fmtPct(v: any): string {
  const n = toNum(v);
  if (n === null) return "";
  return `${Math.round(n)}%`;
}

function fmtScore(v: any): string {
  const n = toNum(v);
  if (n === null) return "";
  return n.toFixed(3);
}

export default function SegmentsTable({
  rows,
  loading,
}: {
  rows: Record<string, any>[];
  loading: boolean;
}) {
  const hasFinal = useMemo(
    () => rows.some((r) => r?.["New Segment Name"] !== undefined),
    [rows]
  );

  const defaultSortKey = useMemo(() => {
    if (rows.some((r) => r?.["Digital Ad Targeting Price (CPM)"] !== undefined))
      return "Digital Ad Targeting Price (CPM)";
    if (rows.some((r) => r?.rank_score !== undefined)) return "rank_score";
    if (hasFinal) return "New Segment Name";
    return "Proposed New Segment Name";
  }, [rows, hasFinal]);

  // ✅ important: keep state stable + sync when default changes between runs
  const [sortKey, setSortKey] = useState<string>(defaultSortKey);
  const [sortDir, setSortDir] = useState<"asc" | "desc">(
    defaultSortKey.includes("Name") ? "asc" : "desc"
  );
  const [pageSize, setPageSize] = useState(50);

  useEffect(() => {
    setSortKey(defaultSortKey);
    setSortDir(defaultSortKey.includes("Name") ? "asc" : "desc");
  }, [defaultSortKey]);

  /** ✅ Columns: show pricing + description, hide "Non Derived Segments utilized" */
  const columns = useMemo(() => {
    if (hasFinal) {
      const base = ["New Segment Name", "Segment Description", ...PRICE_COLS];
      return base.filter((c) => rows.some((r) => r?.[c] !== undefined));
    }

    // fallback for validated rows (no pricing yet)
    const fallback = [
      "Proposed New Segment Name",
      "Segment Description",
      "rank_score",
      "uniqueness_score",
      "Composition Similarity",
      "Closest Cybba Similarity",
    ];
    return fallback.filter((c) => rows.some((r) => r?.[c] !== undefined));
  }, [rows, hasFinal]);

  const sortOptions = useMemo(() => {
    const opts: string[] = [];

    if (rows.some((r) => r?.["New Segment Name"] !== undefined))
      opts.push("New Segment Name");
    if (rows.some((r) => r?.["Proposed New Segment Name"] !== undefined))
      opts.push("Proposed New Segment Name");

    for (const c of PRICE_COLS) {
      if (rows.some((r) => r?.[c] !== undefined)) opts.push(c);
    }
    for (const c of Array.from(SCORE_COLS)) {
      if (rows.some((r) => r?.[c] !== undefined)) opts.push(c);
    }

    return Array.from(new Set(opts));
  }, [rows]);

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = a?.[sortKey];
      const bv = b?.[sortKey];

      const an = toNum(av);
      const bn = toNum(bv);

      if (an !== null && bn !== null) {
        if (an < bn) return sortDir === "asc" ? -1 : 1;
        if (an > bn) return sortDir === "asc" ? 1 : -1;
        return 0;
      }

      if (an !== null && bn === null) return sortDir === "asc" ? -1 : 1;
      if (an === null && bn !== null) return sortDir === "asc" ? 1 : -1;

      const as = String(av ?? "").toLowerCase();
      const bs = String(bv ?? "").toLowerCase();
      if (as < bs) return sortDir === "asc" ? -1 : 1;
      if (as > bs) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  const visible = useMemo(() => sorted.slice(0, pageSize), [sorted, pageSize]);

  // ✅ skeleton only when no rows yet
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

  const nameCellStyle: React.CSSProperties = {
    fontWeight: 900,
    lineHeight: 1.25,
    letterSpacing: "-0.2px",
  };

  const descCellStyle: React.CSSProperties = {
    color: "rgba(255,255,255,0.82)",
    lineHeight: 1.35,
    maxWidth: 760, // ✅ more space now
    whiteSpace: "normal",
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
    whiteSpace: "nowrap",
  };

  return (
    <>
      <div className="tableToolbar">
        <div
          className="muted"
          style={{ display: "inline-flex", gap: 10, alignItems: "center" }}
        >
          <span>
            Showing {Math.min(pageSize, rows.length)} / {rows.length}
          </span>
          {loading ? (
            <span
              style={{ display: "inline-flex", gap: 8, alignItems: "center" }}
            >
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
              {sortOptions.map((k) => (
                <option key={k} value={k}>
                  {COL_LABELS[k] ?? k}
                </option>
              ))}
            </select>
          </label>

          <label className="miniField">
            Dir
            <select
              value={sortDir}
              onChange={(e) => setSortDir(e.target.value as any)}
            >
              <option value="desc">desc</option>
              <option value="asc">asc</option>
            </select>
          </label>

          <label className="miniField">
            Rows
            <select
              value={pageSize}
              onChange={(e) => setPageSize(Number(e.target.value))}
            >
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
                <th key={c} title={c}>
                  {COL_LABELS[c] ?? c}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {visible.map((r, idx) => {
              const key =
                r?.["New Segment Name"] ??
                r?.["Proposed New Segment Name"] ??
                `${idx}`;

              return (
                <tr key={key}>
                  {columns.map((c) => {
                    const v = r?.[c];

                    if (c === "New Segment Name" || c === "Proposed New Segment Name") {
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

                    // ✅ money formatting
                    if (CPM_COLS.has(c)) {
                      return (
                        <td key={c}>
                          <span style={chipStyle}>{fmtMoney(v, 1)}</span>
                        </td>
                      );
                    }
                    if (CPC_COLS.has(c)) {
                      return (
                        <td key={c}>
                          <span style={chipStyle}>{fmtMoney(v, 2)}</span>
                        </td>
                      );
                    }

                    if (PCT_COLS.has(c)) {
                      return (
                        <td key={c}>
                          <span style={chipStyle}>{fmtPct(v)}</span>
                        </td>
                      );
                    }

                    if (SCORE_COLS.has(c)) {
                      return (
                        <td key={c}>
                          <span style={chipStyle}>{fmtScore(v)}</span>
                        </td>
                      );
                    }

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
