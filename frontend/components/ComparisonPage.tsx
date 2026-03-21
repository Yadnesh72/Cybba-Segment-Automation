// frontend/src/pages/ComparisonPage.tsx
import React, { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

type Row = Record<string, any>;

type Match = {
  competitor: string;
  segment_name: string;
  score: number;
};

function getDerivedName(r: Row): string {
  return (
    String(
      r?.["New Segment Name"] ??
        r?.["Proposed New Segment Name"] ??
        r?.["Segment Name"] ??
        ""
    ).trim() || "—"
  );
}

function fmtPct(x: number) {
  return `${Math.round(x * 100)}%`;
}

export default function ComparisonPage({
  rows,
  streaming,
}: {
  rows: Row[];
  streaming: boolean;
}) {
  const PAGE_SIZE = 25;

  const [q, setQ] = useState("");
  const [openKey, setOpenKey] = useState<string | null>(null);

  // controls
  const [minSim, setMinSim] = useState(0.35);
  const [topK, setTopK] = useState(20);

  // pagination (same pattern as Segments page)
  const [page, setPage] = useState(1);
  const [pageAnimKey, setPageAnimKey] = useState(0);

  // cache per segment name
  const [cache, setCache] = useState<
    Record<string, { loading: boolean; error: string | null; matches: Match[] }>
  >({});

  const ourSegments = useMemo(() => {
    const map = new Map<string, { name: string; count: number }>();

    for (const r of rows || []) {
      const name = getDerivedName(r);
      if (!name || name === "—") continue;

      const key = name.toLowerCase();
      const cur = map.get(key);
      if (cur) cur.count += 1;
      else map.set(key, { name, count: 1 });
    }

    let arr = Array.from(map.values());

    const ql = q.trim().toLowerCase();
    if (ql) arr = arr.filter((x) => x.name.toLowerCase().includes(ql));

    arr.sort((a, b) => a.name.localeCompare(b.name));
    return arr;
  }, [rows, q]);

  // reset pagination when filters/search change (same behavior as Segments page)
  useEffect(() => {
    setPage(1);
  }, [q, minSim, topK, rows]);

  const totalPages = useMemo(
    () => Math.max(1, Math.ceil(ourSegments.length / PAGE_SIZE)),
    [ourSegments.length]
  );

  const safePage = Math.min(page, totalPages);

  useEffect(() => {
    if (safePage !== page) setPage(safePage);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [totalPages]);

  const pagedSegments = useMemo(() => {
    const start = (safePage - 1) * PAGE_SIZE;
    return ourSegments.slice(start, start + PAGE_SIZE);
  }, [ourSegments, safePage]);

  function clearCache() {
    setCache({});
    setOpenKey(null);
  }

  async function ensureFetched(name: string) {
    const key = name.toLowerCase();
    const existing = cache[key];
    if (existing && (existing.loading || existing.matches.length || existing.error))
      return;

    setCache((prev) => ({
      ...prev,
      [key]: { loading: true, error: null, matches: [] },
    }));

    try {
      const API_BASE =
        (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8000";

      const url = `${API_BASE}/api/competitor_matches?query=${encodeURIComponent(
        name
      )}&top_k=${encodeURIComponent(String(topK))}&pool=600`;

      const res = await fetch(url);
      const text = await res.text();

      if (!res.ok) {
        throw new Error(`match fetch failed (${res.status}): ${text.slice(0, 160)}`);
      }

      let data: any;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(`Non-JSON response: ${text.slice(0, 160)}`);
      }

      const matchesRaw: Match[] = Array.isArray(data?.matches) ? data.matches : [];
      const matches: Match[] = matchesRaw.filter((m) => (m?.score ?? 0) >= minSim);

      setCache((prev) => ({
        ...prev,
        [key]: { loading: false, error: null, matches },
      }));
    } catch (e: any) {
      setCache((prev) => ({
        ...prev,
        [key]: {
          loading: false,
          error: e?.message ?? "Failed to fetch matches",
          matches: [],
        },
      }));
    }
  }

  const onToggle = (name: string) => {
    const key = name.toLowerCase();
    const willOpen = openKey !== key;

    setOpenKey(willOpen ? key : null);
    if (willOpen) ensureFetched(name);
  };

  return (
    <div className="pageFade">
      <section className="filtersStrip fadeInUp">
        <div className="filtersStripHeader">
          <div className="cardTitle">Comparison</div>
          <div className="cardHint">
            Showing <b>{ourSegments.length}</b> segments
          </div>
        </div>

        <div className="filtersStripBody">
          <div className="field">
            <span>Search</span>
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search your segments..."
            />
          </div>

          <label className="field">
            <span>Min similarity</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={minSim}
              onChange={(e) => {
                setMinSim(Number(e.target.value));
                clearCache();
              }}
            />
          </label>

          <label className="field">
            <span>Top K</span>
            <input
              type="number"
              min={5}
              max={50}
              step={1}
              value={topK}
              onChange={(e) => {
                setTopK(Number(e.target.value));
                clearCache();
              }}
            />
          </label>
        </div>
      </section>

      <section className="card fadeInUp">
        <div className="cardHeader">
          <div className="cardTitle">Your Segment → Competitor Similar Segments</div>

          <div className="muted" style={{ display: "inline-flex", gap: 10, alignItems: "center" }}>
            {streaming ? (
              <>
                <span className="miniSpinner" />
                streaming…
              </>
            ) : (
              <span className="muted">ready</span>
            )}
          </div>
        </div>

        {ourSegments.length === 0 ? (
          <div className="empty" style={{ padding: 14 }}>
            <div className="emptyTitle">No segments available</div>
            <div className="muted" style={{ marginTop: 6 }}>
              Generate segments first, then come back to see competitor similarities.
            </div>
          </div>
        ) : (
          <>
            <div className="cmpAccList" key={pageAnimKey}>
              {pagedSegments.map((s) => {
                const key = s.name.toLowerCase();
                const isOpen = openKey === key;
                const state = cache[key] ?? {
                  loading: false,
                  error: null,
                  matches: [],
                };

                return (
                  <div key={s.name} className="cmpAccItem">
                    <button
                      type="button"
                      className="cmpAccHeader"
                      onClick={() => onToggle(s.name)}
                    >
                      <div className="cmpAccTitle">
                        {s.name}
                      </div>
                      <div className="cmpAccChevron">{isOpen ? "▾" : "▸"}</div>
                    </button>

                    <AnimatePresence initial={false}>
                      {isOpen ? (
                        <motion.div
                          className="cmpAccBody"
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.22, ease: "easeOut" }}
                        >
                          {state.loading ? (
                            <div
                              className="muted"
                              style={{
                                padding: "10px 12px",
                                display: "inline-flex",
                                gap: 8,
                                alignItems: "center",
                              }}
                            >
                              <span className="miniSpinner" />
                              Finding competitor matches…
                            </div>
                          ) : state.error ? (
                            <div className="muted" style={{ padding: "10px 12px" }}>
                              {state.error}
                            </div>
                          ) : state.matches.length === 0 ? (
                            <div className="muted" style={{ padding: "10px 12px" }}>
                              No competitor matches above {fmtPct(minSim)}. Try lowering “Min similarity”.
                            </div>
                          ) : (
                            <div className="cmpMatchGrid">
                              {state.matches.map((m, idx) => (
                                <div
                                  key={`${m.competitor}-${m.segment_name}-${idx}`}
                                  className="cmpMatchCard"
                                >
                                  <div className="cmpMatchTop">
                                    <span className="cmpMatchCompetitor">{m.competitor}</span>
                                    <span className="cmpMatchScore">{fmtPct(m.score)}</span>
                                  </div>
                                  <div className="cmpMatchName">{m.segment_name}</div>
                                </div>
                              ))}
                            </div>
                          )}
                        </motion.div>
                      ) : null}
                    </AnimatePresence>
                  </div>
                );
              })}
            </div>

            {/* ✅ Pagination (same UI as Segments page) */}
            {ourSegments.length > PAGE_SIZE ? (
              <div className="pagerRow">
                <button
                  className="pagerBtn"
                  disabled={safePage <= 1}
                  onClick={() => {
                    setPage((p) => Math.max(1, p - 1));
                    setPageAnimKey((k) => k + 1);
                    setOpenKey(null);
                  }}
                >
                  Prev
                </button>

                <div className="pagerNums">
                  {Array.from({ length: totalPages }, (_, i) => i + 1)
                    .filter(
                      (n) => n === 1 || n === totalPages || Math.abs(n - safePage) <= 2
                    )
                    .reduce<number[]>((acc, n) => {
                      if (acc.length && n - acc[acc.length - 1] > 1) acc.push(-1);
                      acc.push(n);
                      return acc;
                    }, [])
                    .map((n, idx) =>
                      n === -1 ? (
                        <span key={`gap-${idx}`} className="pagerGap">
                          …
                        </span>
                      ) : (
                        <button
                          key={n}
                          className={`pagerNum ${n === safePage ? "pagerNumActive" : ""}`}
                          onClick={() => {
                            setPage(n);
                            setPageAnimKey((k) => k + 1);
                            setOpenKey(null);
                          }}
                        >
                          {n}
                        </button>
                      )
                    )}
                </div>

                <button
                  className="pagerBtn"
                  disabled={safePage >= totalPages}
                  onClick={() => {
                    setPage((p) => Math.min(totalPages, p + 1));
                    setPageAnimKey((k) => k + 1);
                    setOpenKey(null);
                  }}
                >
                  Next
                </button>
              </div>
            ) : null}
          </>
        )}
      </section>

      <style>{`
        .cmpAccList{ display: grid; gap: 10px; }

        .cmpAccItem{
          border-radius: 16px;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(0,0,0,0.10);
          overflow: hidden;
        }

        .cmpAccHeader{
          width: 100%;
          text-align: left;
          background: transparent;
          border: none;
          padding: 12px 12px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
        }

        /* ✅ ensure accordion text is NOT black */
        .cmpAccHeader, .cmpAccTitle, .cmpAccChevron{
          color: rgba(255,255,255,0.90);
        }

        .cmpAccTitle{
          font-weight: 950;
          font-size: 14px;
          line-height: 1.25;
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 10px;
        }

        .cmpAccCount{
          font-size: 12px;
          font-weight: 800;
          opacity: 0.7;
          padding: 2px 8px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
        }

        .cmpAccChevron{
          font-weight: 900;
          opacity: 0.8;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.10);
          background: rgba(255,255,255,0.04);
        }

        .cmpAccBody{
          border-top: 1px solid rgba(255,255,255,0.10);
        }

        .cmpMatchGrid{
          padding: 12px;
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
        }

        @media (max-width: 980px){
          .cmpMatchGrid{ grid-template-columns: 1fr; }
        }

        .cmpMatchCard{
          border-radius: 14px;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.06);
          padding: 10px 12px;
        }

        .cmpMatchTop{
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 10px;
          margin-bottom: 6px;
        }

        .cmpMatchCompetitor{
          font-weight: 900;
          font-size: 12px;
          letter-spacing: 0.10em;
          text-transform: uppercase;
          opacity: 0.85;
        }

        .cmpMatchScore{
          font-weight: 950;
          font-size: 12px;
          opacity: 0.9;
          padding: 2px 8px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.14);
          background: rgba(0,0,0,0.12);
          font-variant-numeric: tabular-nums;
        }

        .cmpMatchName{
          font-weight: 900;
          font-size: 13px;
          line-height: 1.3;
          opacity: 0.95;
          color: rgba(255,255,255,0.92);
        }

        /* ── Light mode overrides ── */
        html[data-theme="light"] .cmpAccItem{
          border-color: rgba(15,23,42,0.10);
          background: rgba(255,255,255,0.80);
        }
        html[data-theme="light"] .cmpAccHeader,
        html[data-theme="light"] .cmpAccTitle,
        html[data-theme="light"] .cmpAccChevron{
          color: rgba(15,23,42,0.88);
        }
        html[data-theme="light"] .cmpAccCount{
          border-color: rgba(15,23,42,0.12);
          background: rgba(15,23,42,0.06);
          color: rgba(15,23,42,0.78);
        }
        html[data-theme="light"] .cmpAccChevron{
          border-color: rgba(15,23,42,0.10);
          background: rgba(15,23,42,0.04);
        }
        html[data-theme="light"] .cmpAccBody{
          border-top-color: rgba(15,23,42,0.08);
        }
        html[data-theme="light"] .cmpMatchCard{
          border-color: rgba(15,23,42,0.10);
          background: rgba(255,255,255,0.85);
        }
        html[data-theme="light"] .cmpMatchScore{
          border-color: rgba(15,23,42,0.12);
          background: rgba(15,23,42,0.05);
          color: rgba(15,23,42,0.88);
        }
        html[data-theme="light"] .cmpMatchName{
          color: rgba(15,23,42,0.88);
        }
        html[data-theme="light"] .cmpMatchCompetitor{
          color: rgba(15,23,42,0.65);
        }
      `}</style>
    </div>
  );
}