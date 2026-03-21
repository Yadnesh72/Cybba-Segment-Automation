import React, { useEffect, useMemo, useState } from "react";
import SegmentsTable from "./SegmentsTable";
import { API_BASE } from "../src/api";

type Row = Record<string, any>;
const PAGE_SIZE = 25;

async function fetchCybbaCatalog(args: { page: number; page_size: number; q?: string }) {
  const qs = new URLSearchParams();
  qs.set("page", String(args.page));
  qs.set("page_size", String(args.page_size));
  if (args.q) qs.set("q", args.q);

  const res = await fetch(`${API_BASE}/api/catalog/cybba?${qs.toString()}`);
  if (!res.ok) throw new Error(`Catalog request failed (${res.status})`);
  return res.json();
}

export default function CybbaSegmentsPage({ streaming }: { streaming: boolean }) {
  const [q, setQ] = useState("");
  const [page, setPage] = useState(1);

  const [rawRows, setRawRows] = useState<Row[]>([]);
  const [total, setTotal] = useState(0);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const totalPages = useMemo(() => Math.max(1, Math.ceil((total || 0) / PAGE_SIZE)), [total]);

  // ✅ map backend shape -> SegmentsTable expected shape
  const rows = useMemo(() => {
    return rawRows.map((r) => ({
      ...r,
      // SegmentsTable wants "New Segment Name" or "Proposed New Segment Name"
      "New Segment Name": r["New Segment Name"] ?? r["Segment Name"] ?? "",
      // Segment Description already matches your backend column
      "Segment Description": r["Segment Description"] ?? "",
    }));
  }, [rawRows]);

  async function load(nextPage: number, nextQ: string) {
    setLoading(true);
    setErr(null);
    try {
      const res = await fetchCybbaCatalog({ page: nextPage, page_size: PAGE_SIZE, q: nextQ });
      setRawRows(res.rows ?? []);
      setTotal(Number(res.total ?? 0));
      setPage(Number(res.page ?? nextPage));
    } catch (e: any) {
      setErr(e?.message ?? "Failed to load Cybba catalog");
      setRawRows([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }

  // initial load
  useEffect(() => {
    load(1, "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="pageFade">
      <section className="filtersStrip fadeInUp">
        <div className="filtersStripHeader">
          <div className="cardTitle">Cybba segments</div>
          <div className="cardHint">
            Showing <b>{rows.length}</b> / {total}
          </div>
        </div>

        <div className="filtersStripBody" style={{ gridTemplateColumns: "2fr auto 1fr" }}>
          <div className="field">
            <span>Search</span>
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search Segment Name / Description…"
              disabled={loading}
            />
          </div>

          <button className="btnGhost" type="button" onClick={() => load(1, q)} disabled={loading}>
            {loading ? "Loading…" : "Search"}
          </button>

          <div className="field">
            <span>Status</span>
            <div className="statusPill">
              {streaming ? (
                <>
                  <span className="miniSpinner" />
                  <span>streaming…</span>
                </>
              ) : (
                <span className="muted">ready</span>
              )}
            </div>
          </div>
        </div>
      </section>

      {err ? (
        <div className="toast toastShow">
          <div className="toastTitle">Catalog error</div>
          <div className="toastBody">{err}</div>
        </div>
      ) : null}

      <section className="card fadeInUp">
        <div className="cardHeader">
          <div className="cardTitle">Existing Cybba segments</div>
        </div>

        <SegmentsTable rows={rows} loading={loading} />

        {totalPages > 1 ? (
          <div className="pagerRow">
            <button className="pagerBtn" disabled={loading || page <= 1} onClick={() => load(page - 1, q)}>
              Prev
            </button>

            <div className="pagerNums">
              <span className="muted" style={{ fontWeight: 800 }}>
                Page {page} / {totalPages}
              </span>
            </div>

            <button className="pagerBtn" disabled={loading || page >= totalPages} onClick={() => load(page + 1, q)}>
              Next
            </button>
          </div>
        ) : null}
      </section>
    </div>
  );
}