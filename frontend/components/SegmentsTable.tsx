import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom";

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

/** UI-only abbreviations */
const COL_LABELS: Record<string, string> = {
  "New Segment Name": "Segment",
  "Proposed New Segment Name": "Segment",
  "Segment Description": "Description",

  "Digital Ad Targeting Price (CPM)": "D-CPM",
  "Content Marketing Price (CPM)": "C-CPM",
  "TV Targeting Price (CPM)": "TV-CPM",
  "Cost Per Click": "CPC",
  "Programmatic % of Media": "Prog%",
  "CPM Cap": "Cap",
  "Advertiser Direct % of Media": "Direct%",
};

/** Header tooltip text */
const COL_TIPS: Record<string, { title: string; body: string }> = {
  "New Segment Name": {
    title: "Segment",
    body: "Generated segment name. Primary identifier used for sorting and merging priced results.",
  },
  "Proposed New Segment Name": {
    title: "Segment",
    body: "Generated segment name (validated stage). This becomes the final segment name after pricing/export.",
  },
  "Segment Description": {
    title: "Description",
    body: "Short explanation of the audience definition used to generate and price the segment.",
  },

  "Digital Ad Targeting Price (CPM)": {
    title: "Digital Ad Targeting (CPM)",
    body: "Estimated CPM for digital audience targeting inventory for this segment.",
  },
  "Content Marketing Price (CPM)": {
    title: "Content Marketing (CPM)",
    body: "Estimated CPM for content marketing / native inventory for this segment.",
  },
  "TV Targeting Price (CPM)": {
    title: "TV Targeting (CPM)",
    body: "Estimated CPM for addressable / targeted TV inventory for this segment.",
  },
  "Cost Per Click": {
    title: "Cost Per Click (CPC)",
    body: "Estimated average CPC for campaigns targeting this segment.",
  },
  "Programmatic % of Media": {
    title: "Programmatic %",
    body: "Estimated share of media expected to be programmatic for this segment.",
  },
  "CPM Cap": {
    title: "CPM Cap",
    body: "Estimated recommended CPM cap (upper bound) to control average CPM for this segment.",
  },
  "Advertiser Direct % of Media": {
    title: "Advertiser Direct %",
    body: "Estimated share of media expected to be bought direct (non-programmatic) for this segment.",
  },
};

function toNum(v: any): number | null {
  if (v === null || v === undefined || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

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

/** ---------------- Tooltip (cursor-follow, portal, no-flicker) ---------------- */
type TipState = {
  title: string;
  body: string;
  x: number;
  y: number;
} | null;

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function TooltipPortal({ tip }: { tip: TipState }) {
  const [mounted, setMounted] = useState(false);
  const [pos, setPos] = useState({ left: 0, top: 0 });
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    setMounted(true);
    return () => {
      setMounted(false);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  useEffect(() => {
    if (!tip) return;

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const pad = 12;
      const gap = 14;

      const vw = window.innerWidth;
      const vh = window.innerHeight;

      const width = 340;
      const estH = 96;

      let left = tip.x + gap;
      let top = tip.y + gap;

      left = clamp(left, pad, vw - width - pad);
      if (top + estH > vh - pad) {
        top = clamp(tip.y - gap - estH, pad, vh - estH - pad);
      }

      setPos({ left, top });
    });

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [tip?.x, tip?.y, tip?.title, tip?.body]);

  if (!mounted) return null;

  const node = (
    <div
      className={`kpiTip ${tip ? "kpiTipShow" : ""}`}
      style={{ left: pos.left, top: pos.top }}
      role="tooltip"
      aria-hidden={!tip}
    >
      <div className="kpiTipTitle">{tip?.title ?? ""}</div>
      <div className="kpiTipBody">{tip?.body ?? ""}</div>

      <style>{`
        .kpiTip{
          position: fixed;
          z-index: 999999;
          width: 340px;
          max-width: min(340px, calc(100vw - 24px));
          padding: 12px 14px;
          border-radius: 14px;
          border: 1px solid rgba(255,255,255,0.10);
          background: rgba(12, 16, 28, 0.55);
          backdrop-filter: blur(14px);
          -webkit-backdrop-filter: blur(14px);
          box-shadow:
            0 18px 50px rgba(0,0,0,0.42),
            0 0 0 1px rgba(255,255,255,0.04) inset;
          pointer-events: none;

          opacity: 0;
          transform: translateY(6px) scale(0.985);
          transition: opacity 160ms ease, transform 180ms ease;
        }
        .kpiTipShow{
          opacity: 1;
          transform: translateY(0px) scale(1);
        }
        .kpiTipTitle{
          font-weight: 900;
          font-size: 13px;
          letter-spacing: -0.2px;
          margin-bottom: 6px;
          color: rgba(255,255,255,0.92);
        }
        .kpiTipBody{
          font-size: 12.5px;
          line-height: 1.35;
          color: rgba(255,255,255,0.78);
        }
        .kpiTip::before, .kpiTip::after{
          content: none !important;
          display: none !important;
        }

        /* Header hover target */
        .thHover{
          display: inline-flex;
          align-items: center;
          gap: 6px;
          cursor: help;
          user-select: none;
          color: rgba(255,255,255,0.88);
        }
        .thHover:hover{
          color: rgba(255,255,255,0.98);
          text-decoration: none;
        }
      `}</style>
    </div>
  );

  return ReactDOM.createPortal(node, document.body);
}

export default function SegmentsTable({
  rows,
  loading,
}: {
  rows: Record<string, any>[];
  loading: boolean;
}) {
  const [tip, setTip] = useState<TipState>(null);
  const showTimerRef = useRef<number | null>(null);
  const lastMouseRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  const clearShowTimer = () => {
    if (showTimerRef.current) {
      window.clearTimeout(showTimerRef.current);
      showTimerRef.current = null;
    }
  };

  const requestShow = (title: string, body: string) => {
    clearShowTimer();
    showTimerRef.current = window.setTimeout(() => {
      setTip({ title, body, x: lastMouseRef.current.x, y: lastMouseRef.current.y });
    }, 120);
  };

  const hide = () => {
    clearShowTimer();
    setTip(null);
  };

  useEffect(() => {
    return () => clearShowTimer();
  }, []);

  // if we have "New Segment Name" we’re in final/priced shape; otherwise validated shape
  const nameKey = useMemo(() => {
    return rows.some((r) => r?.["New Segment Name"] != null)
      ? "New Segment Name"
      : "Proposed New Segment Name";
  }, [rows]);

  const columns = useMemo(() => {
    // Always show pricing columns from the start (blank until present)
    return [nameKey, "Segment Description", ...PRICE_COLS] as string[];
  }, [nameKey]);

  const defaultSortKey = useMemo(() => "Digital Ad Targeting Price (CPM)", []);
  const [sortKey, setSortKey] = useState<string>(defaultSortKey);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [pageSize, setPageSize] = useState(50);

  useEffect(() => {
    setSortKey(defaultSortKey);
    setSortDir("desc");
  }, [defaultSortKey]);

  const sortOptions = useMemo(() => {
    return [nameKey, ...PRICE_COLS] as string[];
  }, [nameKey]);

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

      // put numeric values first
      if (an !== null && bn === null) return -1;
      if (an === null && bn !== null) return 1;

      const as = String(av ?? "").toLowerCase();
      const bs = String(bv ?? "").toLowerCase();
      if (as < bs) return sortDir === "asc" ? -1 : 1;
      if (as > bs) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  const visible = useMemo(() => sorted.slice(0, pageSize), [sorted, pageSize]);

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
    maxWidth: 760,
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
      <TooltipPortal tip={tip} />

      <div className="tableToolbar">
        <div className="muted" style={{ display: "inline-flex", gap: 10, alignItems: "center" }}>
          <span>
            Showing {Math.min(pageSize, rows.length)} / {rows.length}
          </span>
          {loading ? (
            <span style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
              <span className="btnSpinner btnSpinnerShow" style={{ display: "inline-block" }} />
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
              {columns.map((c) => {
                const tipDef = COL_TIPS[c];
                const label = COL_LABELS[c] ?? c;

                // If we don't have a tooltip definition, just render the label normally
                if (!tipDef) {
                  return (
                    <th key={c} title={c}>
                      {label}
                    </th>
                  );
                }

                // Hover target = the header label itself
                return (
                  <th key={c} title={c}>
                    <span
                      className="thHover"
                      onMouseEnter={(e) => {
                        lastMouseRef.current = { x: e.clientX, y: e.clientY };
                        requestShow(tipDef.title, tipDef.body);
                      }}
                      onMouseMove={(e) => {
                        lastMouseRef.current = { x: e.clientX, y: e.clientY };
                        setTip((prev) => (prev ? { ...prev, x: e.clientX, y: e.clientY } : prev));
                      }}
                      onMouseLeave={() => hide()}
                    >
                      {label}
                    </span>
                  </th>
                );
              })}
            </tr>
          </thead>

          <tbody>
            {visible.map((r, idx) => {
              const key = String(r?.[nameKey] ?? idx);

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

                    // money formatting
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