// frontend/src/components/AnalyticsPage.tsx
import React, { useMemo, useRef, useState, useEffect, useCallback } from "react";
import { useTheme } from "../src/useTheme";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { motion, AnimatePresence } from "framer-motion";
import { API_BASE } from "../src/api";
import html2canvas from "html2canvas";



type Row = Record<string, any>;

function normKey(k: string) {
  return k.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function getByAnyKey(r: Row, wanted: string[]) {
  for (const k of wanted) {
    if ((r as any)?.[k] != null && (r as any)[k] !== "") return (r as any)[k];
  }

  const wantedNorm = new Set(wanted.map(normKey));
  for (const k of Object.keys(r || {})) {
    if (wantedNorm.has(normKey(k))) {
      const v = (r as any)[k];
      if (v != null && v !== "") return v;
    }
  }

  return null;
}

function toNum(v: any): number | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "number") return Number.isFinite(v) ? v : null;

  const s = String(v).trim();
  if (!s) return null;

  const cleaned = s.replace(/,/g, "").replace(/%/g, "");
  const n = Number(cleaned);
  return Number.isFinite(n) ? n : null;
}

function fmt(n: number | null) {
  if (n == null) return "—";
  if (Math.abs(n) >= 100) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(1);
  return n.toFixed(2);
}

function median(nums: number[]) {
  if (!nums.length) return null;
  const a = [...nums].sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

function minmax(nums: number[]) {
  if (!nums.length) return { min: null as number | null, max: null as number | null };
  return { min: Math.min(...nums), max: Math.max(...nums) };
}

function quantile(nums: number[], p: number) {
  if (!nums.length) return null;
  const a = [...nums].sort((x, y) => x - y);
  const idx = Math.min(a.length - 1, Math.max(0, Math.floor(p * (a.length - 1))));
  return a[idx];
}

function pearson(xs: number[], ys: number[]) {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return null;

  let sx = 0,
    sy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;

  let num = 0,
    dx = 0,
    dy = 0;
  for (let i = 0; i < n; i++) {
    const vx = xs[i] - mx;
    const vy = ys[i] - my;
    num += vx * vy;
    dx += vx * vx;
    dy += vy * vy;
  }
  const den = Math.sqrt(dx * dy);
  if (!Number.isFinite(den) || den === 0) return null;
  return num / den;
}

function pct(part: number, total: number) {
  const t = total || 1;
  return Math.round((part / t) * 100);
}

function bucketize(values: number[], bins = 18) {
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return [{ x: min, y: values.length, lo: min, hi: max }];

  const step = (max - min) / bins;
  const counts = Array.from({ length: bins }, () => 0);

  for (const v of values) {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / step)));
    counts[idx] += 1;
  }

  return counts.map((c, i) => ({
    x: min + step * (i + 0.5),
    y: c,
    lo: min + step * i,
    hi: min + step * (i + 1),
  }));
}

/** ---------------- field pickers ---------------- */

function getSources(r: Row): string[] {
  const candidates = [
    r?.underived_segments,
    r?.underived,
    r?.components,
    r?.component_segments,
    r?.source_segments,
    r?.["Underived Segments"],
    r?.["Underived Segment Names"],
    r?.["Component Segments"],
    r?.["Source Segments"],
    r?.["Non Derived Segments utilized"],
    r?.["Non-Derived Segments utilized"],
    r?.["Non Derived Segments Utilized"],
  ];

  for (const c of candidates) {
    if (!c) continue;

    if (Array.isArray(c)) return c.map((x) => String(x).trim()).filter(Boolean);

    if (typeof c === "string") {
      const s = c.trim();
      if (!s) continue;

      if ((s.startsWith("[") && s.endsWith("]")) || (s.startsWith("{") && s.endsWith("}"))) {
        try {
          const parsed = JSON.parse(s);
          if (Array.isArray(parsed)) return parsed.map((x) => String(x).trim()).filter(Boolean);
        } catch {}
      }

      const parts = s.split(/[\n,;|]+/g).map((x) => x.trim()).filter(Boolean);
      if (parts.length) return parts;
    }
  }

  return [];
}
function piePctLabel(props: any) {
  const { percent, midAngle, innerRadius, outerRadius, cx, cy } = props;

  const p = Math.round((percent ?? 0) * 100);
  if (p <= 0) return null; // only hide true 0%

  const RADIAN = Math.PI / 180;

  const radius = innerRadius
    ? innerRadius + (outerRadius - innerRadius) * 0.55
    : outerRadius * 0.7;

  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  return (
    <text
      x={x}
      y={y}
      textAnchor="middle"
      dominantBaseline="central"
      fill="rgba(255,255,255,0.82)"
      style={{
        fontSize: 10,
        fontWeight: 800,
        letterSpacing: "0.2px",
        pointerEvents: "none",
      }}
    >
      {p}%
    </text>
  );
}

function pieOuterLabel(props: any) {
  const { cx, cy, midAngle, outerRadius, percent } = props;

  const p = Math.round((percent ?? 0) * 100);
  if (p <= 0) return null; // only hide true 0%

  const RAD = Math.PI / 180;

  const x0 = cx + outerRadius * Math.cos(-midAngle * RAD);
  const y0 = cy + outerRadius * Math.sin(-midAngle * RAD);

  const r1 = outerRadius + 12;
  const x1 = cx + r1 * Math.cos(-midAngle * RAD);
  const y1 = cy + r1 * Math.sin(-midAngle * RAD);

  const isRight = x1 >= cx;
  const x2 = x1 + (isRight ? 10 : -10);
  const y2 = y1;

  const textAnchor = isRight ? "start" : "end";

  return (
    <g>
      <path
        d={`M${x0},${y0} L${x1},${y1} L${x2},${y2}`}
        stroke="rgba(255,255,255,0.28)"
        strokeWidth={1}
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <text
        x={x2 + (isRight ? 6 : -6)}
        y={y2}
        textAnchor={textAnchor}
        dominantBaseline="central"
        fill="rgba(255,255,255,0.82)"
        style={{
          fontSize: 10,
          fontWeight: 800,
          letterSpacing: "0.2px",
          pointerEvents: "none",
        }}
      >
        {p}%
      </text>
    </g>
  );
}

function getName(r: Row): string {
  return String(
    r?.["New Segment Name"] ?? r?.["Proposed New Segment Name"] ?? r?.["Segment Name"] ?? ""
  ).trim();
}

function getL1FromNewSegmentName(r: Row): string | null {
  const name = getName(r);
  if (!name) return null;

  const raw = name.trim();

  // If it's in canonical format: "Cybba > L1 > L2 > Leaf"
  if (raw.includes(">")) {
    const parts = raw
      .split(">")
      .map((x) => x.trim())
      .filter(Boolean);

    // parts[0] is provider ("Cybba"), parts[1] is L1
    if (parts.length >= 2) {
      const maybeProvider = parts[0].toLowerCase();
      const l1 = maybeProvider === "cybba" ? parts[1] : parts[0];

      const low = l1.toLowerCase();
      if (low === "b2b" || low.includes("b2b")) return "B2B";
      if (low.includes("functional")) return "Functional";
      if (low.includes("purchase")) return "Purchase Audience";
      if (low.includes("behavior")) return "Behavioral";
      if (low.includes("demographic")) return "Demographic";
      if (low.includes("interest")) return "Interest";
      if (low.includes("intent")) return "Intent";
      if (low.includes("geo") || low.includes("location")) return "Geo";

      return l1;
    }
  }

  // Fallback: your old generic splitting behavior
  const pipeIdx = raw.indexOf("|");
  const cleaned = pipeIdx >= 0 ? raw.slice(pipeIdx + 1).trim() : raw;

  const parts = cleaned
    .split(/\s*(?:>|›|\||\/|:|—|-|→)\s*/g)
    .map((x) => x.trim())
    .filter(Boolean);

  if (!parts.length) return null;

  const l1 = parts[0];
  const low = l1.toLowerCase();
  if (low === "b2b" || low.includes("b2b")) return "B2B";
  if (low.includes("functional")) return "Functional";
  if (low.includes("purchase")) return "Purchase Audience";
  if (low.includes("behavior")) return "Behavioral";
  if (low.includes("demographic")) return "Demographic";
  if (low.includes("interest")) return "Interest";
  if (low.includes("intent")) return "Intent";
  if (low.includes("geo") || low.includes("location")) return "Geo";

  return l1;
}

function getL1Category(r: Row): string | null {
  // We try a bunch of likely headers for "L1" / category level 1.
  // If your export uses a different header, add it here.
  const v = getByAnyKey(r, [
    // common & clean
    "L1",
    "L1 Category",
    "Category L1",
    "Taxonomy L1",
    "taxonomy_l1",
    "l1",
    "l1_category",

    // “generated segments” style
    "Generated L1",
    "Generated Category",
    "Generated Category L1",
    "New Segment L1",
    "New Segment Category",
    "New Segment Category L1",

    // occasionally stored as “Segment Type”
    "Segment L1",
    "Segment Type L1",
    "Segment Category L1",

    // fallbacks you might have in your sheet
    "Taxonomy",
    "Taxonomy Category",
    "Segment Category",
    "Category",
  ]);

  if (v == null) return null;

  const s = String(v).trim();
  if (!s) return null;

  // normalize some common patterns (light-touch)
  const low = s.toLowerCase();
  if (low === "b2b" || low.includes("b2b")) return "B2B";
  if (low.includes("functional")) return "Functional";
  if (low.includes("demographic")) return "Demographic";
  if (low.includes("interest")) return "Interest";
  if (low.includes("intent")) return "Intent";
  if (low.includes("purchase")) return "Purchase";
  if (low.includes("behavior")) return "Behavioral";
  if (low.includes("location") || low.includes("geo")) return "Geo";

  return s;
}

function pickPrice(r: Row, which: "d" | "c" | "tv" | "cpc"): number | null {
  const v =
    which === "d"
      ? getByAnyKey(r, ["Digital Ad Targeting Price (CPM)", "D-CPM", "d_cpm", "D CPM", "dcpm"])
      : which === "c"
      ? getByAnyKey(r, ["Content Marketing Price (CPM)", "C-CPM", "c_cpm", "C CPM", "ccpm"])
      : which === "tv"
      ? getByAnyKey(r, ["TV Targeting Price (CPM)", "TV-CPM", "tv_cpm", "TV CPM", "tvcpm"])
      : getByAnyKey(r, ["Cost Per Click", "CPC", "cpc", "c_p_c"]);

  return toNum(v);
}

// Make these extra-robust, since different exports name them wildly.
function pickRank(r: Row): number | null {
  return toNum(
    getByAnyKey(r, [
      "rank_score",
      "Rank Score",
      "Rank",
      "rank",
      "Rank score",
      "RankScore",
      "rankscore",
      "rank_score_normalized",
      "Rank (1-100)",
      "Rank (0-1)",
      "RankValue",
      "Rank Value",
      "Segment Rank",
      "Score Rank",
      "score_rank",
    ])
  );
}

function pickUniq(r: Row): number | null {
  return toNum(
    getByAnyKey(r, [
      "uniqueness_score",
      "Uniqueness Score",
      "Uniq",
      "uniqueness",
      "Uniqueness",
      "UniquenessScore",
      "uniqscore",
      "Uniqueness (0-1)",
      "Uniqueness (1-100)",
      "Unique Score",
      "UniquenessValue",
      "Uniqueness Value",
      "Segment Uniqueness",
      "score_uniqueness",
    ])
  );
}

/** ---------------- labels ---------------- */

function cleanSourceLabel(s: string) {
  const raw = String(s ?? "").trim();
  if (!raw) return "—";

  const pipe = raw.indexOf("|");
  const noId = pipe >= 0 ? raw.slice(pipe + 1).trim() : raw;

  const parts = noId.split(">").map((x) => x.trim()).filter(Boolean);
  const compact = parts.length >= 2 ? `${parts[parts.length - 2]} › ${parts[parts.length - 1]}` : noId;

  const max = 44;
  if (compact.length <= max) return compact;
  return compact.slice(0, max - 1).trimEnd() + "…";
}

/** ---------------- tooltip base ---------------- */

function GlassTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;

  return (
    <div
      style={{
        padding: "10px 12px",
        borderRadius: 14,
        border: "1px solid rgba(255,255,255,0.12)",
        background: "rgba(12, 16, 28, 0.55)",
        backdropFilter: "blur(14px)",
        WebkitBackdropFilter: "blur(14px)",
        boxShadow: "0 18px 50px rgba(0,0,0,0.42), 0 0 0 1px rgba(255,255,255,0.04) inset",
        color: "rgba(255,255,255,0.9)",
        fontSize: 12.5,
        lineHeight: 1.35,
        maxWidth: 440,
      }}
    >
      <div style={{ fontWeight: 950, marginBottom: 6 }}>{label ?? "Details"}</div>
      {payload.map((p: any, i: number) => (
        <div
          key={`${p.dataKey ?? p.name ?? i}`}
          style={{ display: "flex", justifyContent: "space-between", gap: 12 }}
        >
          <span style={{ opacity: 0.75, fontWeight: 800 }}>{p.name ?? p.dataKey}</span>
          <span style={{ fontWeight: 900, fontVariantNumeric: "tabular-nums" }}>
            {typeof p.value === "number" ? fmt(p.value) : String(p.value)}
          </span>
        </div>
      ))}
    </div>
  );
}

/** ---------------- follow-cursor tooltip (NO top-left flash) ---------------- */

type Pt = { x: number; y: number } | null;

function useFollowTooltip() {
  const [pt, setPt] = useState<Pt>(null);
  const raf = useRef<number | null>(null);
  const next = useRef<Pt>(null);

  const flush = useCallback(() => {
    raf.current = null;
    setPt(next.current);
  }, []);

  const onMove = useCallback(
    (e: any) => {
      // Most recharts events provide activeCoordinate (cartesian), but pies sometimes only have chartX/Y.
      const c = e?.activeCoordinate;
      const x = c?.x ?? e?.chartX ?? e?.x;
      const y = c?.y ?? e?.chartY ?? e?.y;

      if (x == null || y == null) return;

      next.current = { x, y };
      if (raf.current == null) raf.current = requestAnimationFrame(flush);
    },
    [flush]
  );

  const onLeave = useCallback(() => {
    next.current = null;
    if (raf.current != null) cancelAnimationFrame(raf.current);
    raf.current = null;
    setPt(null);
  }, []);

  useEffect(() => {
    return () => {
      if (raf.current != null) cancelAnimationFrame(raf.current);
    };
  }, []);

  return { pt, onMove, onLeave };
}

// This tooltip animates “from cursor” because it’s positioned at cursor + offset.
// The key is: we ONLY render it when pt exists; otherwise it never flashes at (0,0).
function CursorTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98, y: -6 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.98, y: -6 }}
      transition={{ duration: 0.10, ease: "easeOut" }}
      style={{ pointerEvents: "none", willChange: "transform, opacity" }}
    >
      <GlassTooltip active={active} payload={payload} label={label} />
    </motion.div>
  );
}

function CursorOnlyTooltip({ pt, content }: { pt: { x: number; y: number } | null; content: any }) {
  return (
    <Tooltip
      content={content}
      position={pt ? { x: pt.x + 14, y: pt.y - 12 } : { x: -9999, y: -9999 }}
      allowEscapeViewBox={{ x: true, y: true }}
      wrapperStyle={{ outline: "none", pointerEvents: "none" }}
    />
  );
}

/** ---------------- UI blocks ---------------- */

function CardEnter({
  children,
  delay = 0,
  k,
  onClick,
}: {
  children: React.ReactNode;
  delay?: number;
  k?: string | number;
  onClick?: () => void;
}) {
  return (
    <motion.section
      key={k}
      className={`card ${onClick ? "cardClickable" : ""}`}
      initial={{ opacity: 0, y: 10, scale: 0.995 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.28, ease: "easeOut", delay }}
      onClick={onClick}
    >
      {children}
    </motion.section>
  );
}

function MetricTabs({
  value,
  onChange,
  disabled,
}: {
  value: "d" | "c" | "tv" | "cpc";
  onChange: (v: "d" | "c" | "tv" | "cpc") => void;
  disabled?: boolean;
}) {
  const items: { id: "d" | "c" | "tv" | "cpc"; label: string }[] = [
    { id: "d", label: "D-CPM" },
    { id: "c", label: "C-CPM" },
    { id: "tv", label: "TV-CPM" },
    { id: "cpc", label: "CPC" },
  ];

  return (
    <div className={`segTabs ${disabled ? "segTabsDisabled" : ""}`} role="tablist" aria-label="Price metric">
      {items.map((it) => (
        <button
          key={it.id}
          type="button"
          role="tab"
          aria-selected={value === it.id}
          className={`segTab ${value === it.id ? "segTabActive" : ""}`}
          onClick={() => onChange(it.id)}
          disabled={disabled}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
      <span className="muted" style={{ fontWeight: 800 }}>
        {label}
      </span>
      <span style={{ fontWeight: 950, fontVariantNumeric: "tabular-nums" }}>{value}</span>
    </div>
  );
}

function formatAiText(s: string) {
  const raw = String(s ?? "").trim();
  if (!raw) return "";

  return raw
    // remove markdown bold markers but keep the text
    .replace(/\*\*(.*?)\*\*/g, "$1")
    // normalize bullets like "* " into "• "
    .replace(/^\s*\*\s+/gm, "• ")
    // normalize double newlines a bit
    .replace(/\n{3,}/g, "\n\n");
}

/** ---------------- LLM Insights (frontend stub) ---------------- */

type LLMState = {
  loading: boolean;
  text: string;
  error: string | null;
};
async function captureChartBase64(el: HTMLElement): Promise<string> {
  const canvas = await html2canvas(el, { backgroundColor: null, scale: 2 });
  const dataUrl = canvas.toDataURL("image/png");
  return dataUrl.replace(/^data:image\/png;base64,/, "");
}
function useLLMInsights() {
  const [state, setState] = useState<LLMState>({ loading: false, text: "", error: null });

  const run = useCallback(async (payload: any) => {
    setState({ loading: true, text: "", error: null });

    try {
      // You implement this endpoint. Return { text: string }.
      // Example server prompt should summarize chart + suggest improvements.
      const res = await fetch(`${API_BASE}/api/analytics/insights`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const msg = `LLM request failed (${res.status})`;
        setState({ loading: false, text: "", error: msg });
        return;
      }

      const json = await res.json();
      setState({ loading: false, text: String(json?.text ?? ""), error: null });
    } catch (e: any) {
      setState({ loading: false, text: "", error: e?.message ?? "LLM request failed" });
    }
  }, []);

  const clear = useCallback(() => setState({ loading: false, text: "", error: null }), []);

  return { state, run, clear };
}

/** ---------------- Modal ---------------- */

type ChartId =
  | "priceBuckets"
  | "priceDist"
  | "categoryDist"
  | "scatter"
  | "topReused"
  | "reuseDepth"
  | null;

function ChartModal({
  open,
  title,
  subtitle,
  onClose,
  children,
  stats,
  takeaway,
  llm,
  onAskAI,
  chartKey,
  sidebar,
}: {
  open: boolean;
  title: string;
  subtitle?: string;
  takeaway?: string;
  onClose: () => void;
  children: React.ReactNode;
  stats?: { label: string; value: string }[];
  llm: LLMState;
  onAskAI: () => void;
  chartKey: string | null;
  sidebar?: React.ReactNode; // ✅ add
}) {
  const { isLight } = useTheme();
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          className="chartModalRoot"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          style={{ position: "fixed", inset: 0, zIndex: 999999 }}
          onClick={onClose}
          role="dialog"
          aria-modal="true"
        >
          <motion.div
            className="chartModalBackdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: "absolute",
              inset: 0,
              background: isLight ? "rgba(15,23,42,0.30)" : "rgba(0,0,0,0.55)",
              backdropFilter: "blur(10px)",
              WebkitBackdropFilter: "blur(10px)",
            }}
          />

          <motion.div
            className="chartModalCard"
            initial={{ opacity: 0, y: 18, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 14, scale: 0.985 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            style={{
              position: "relative",
              width: "min(1080px, calc(100vw - 28px))",
              margin: "min(52px, 6vh) auto",
              borderRadius: 18,
              border: isLight ? "1px solid rgba(15,23,42,0.10)" : "1px solid rgba(255,255,255,0.12)",
              background: isLight ? "rgba(255,255,255,0.96)" : "rgba(12, 16, 28, 0.72)",
              boxShadow: isLight ? "0 24px 80px rgba(15,23,42,0.20)" : "0 30px 110px rgba(0,0,0,0.55)",
              overflow: "hidden",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ padding: 16, borderBottom: isLight ? "1px solid rgba(15,23,42,0.08)" : "1px solid rgba(255,255,255,0.08)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start" }}>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: 950, fontSize: 16 }}>{title}</div>
                  {subtitle ? <div className="muted" style={{ marginTop: 6 }}>{subtitle}</div> : null}
                  {takeaway ? (
                    <div
                      style={{
                        marginTop: 10,
                        padding: "10px 12px",
                        borderRadius: 14,
                        border: isLight ? "1px solid rgba(15,23,42,0.08)" : "1px solid rgba(255,255,255,0.10)",
                        background: isLight ? "rgba(15,23,42,0.04)" : "rgba(255,255,255,0.04)",
                        color: isLight ? "rgba(15,23,42,0.80)" : "rgba(255,255,255,0.86)",
                        fontSize: 12.5,
                        lineHeight: 1.35,
                      }}
                    >
                      <span className="muted" style={{ fontWeight: 900 }}>Takeaway:</span>{" "}
                      <span style={{ fontWeight: 850 }}>{takeaway}</span>
                    </div>
                  ) : null}
                </div>

                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <button
                    className="btnGhost"
                    onClick={onAskAI}
                    type="button"
                    style={{ borderRadius: 12, padding: "8px 10px" }}
                    aria-label="Ask AI"
                    title="Ask AI"
                    disabled={llm.loading}
                  >
                    {llm.loading ? "Thinking…" : "Ask AI"}
                  </button>
                  <button
                    className="btnGhost"
                    onClick={onClose}
                    type="button"
                    style={{ borderRadius: 12, padding: "8px 10px" }}
                    aria-label="Close"
                    title="Close"
                  >
                    ✕
                  </button>
                </div>
              </div>

              {stats?.length && !sidebar ? (
                <div
                  style={{
                    marginTop: 12,
                    display: "grid",
                    gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
                    gap: 10,
                  }}
                >
                  {stats.slice(0, 6).map((s) => (
                    <div
                      key={s.label}
                      style={{
                        borderRadius: 14,
                        border: isLight ? "1px solid rgba(15,23,42,0.08)" : "1px solid rgba(255,255,255,0.10)",
                        background: isLight ? "rgba(15,23,42,0.04)" : "rgba(255,255,255,0.04)",
                        padding: "10px 12px",
                      }}
                    >
                      <div className="muted" style={{ fontWeight: 850, fontSize: 12 }}>{s.label}</div>
                      <div style={{ fontWeight: 950, marginTop: 6 }}>{s.value}</div>
                    </div>
                  ))}
                </div>
              ) : null}

              {/* LLM panel */}
              {llm.error || llm.text ? (
                <div
                  style={{
                    marginTop: 12,
                    padding: "10px 12px",
                    borderRadius: 14,
                    border: isLight ? "1px solid rgba(15,23,42,0.08)" : "1px solid rgba(255,255,255,0.10)",
                    background: isLight ? "rgba(15,23,42,0.03)" : "rgba(255,255,255,0.04)",
                    color: isLight ? "rgba(15,23,42,0.80)" : "rgba(255,255,255,0.86)",
                    fontSize: 12.5,
                    lineHeight: 1.45,
                    whiteSpace: "pre-wrap",
                  }}
                >
                  <div className="muted" style={{ fontWeight: 900, marginBottom: 6 }}>
                    AI insights
                  </div>
                  {llm.error ? (
                    <div style={{ opacity: 0.9 }}>
                      <b>Couldn’t fetch insights.</b> {llm.error}
                      <div className="muted" style={{ marginTop: 6 }}>
                        Implement <code>/api/analytics/insights</code> to enable this.
                      </div>
                    </div>
                  ) : (
                    formatAiText(llm.text)
                  )}
                </div>
              ) : null}
            </div>

            {/* BODY */}
            <AnimatePresence mode="wait">
            {/* BODY */}
            <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
            style={{
                padding: 14,
            }}
            >
            {/* ONE container for BOTH metrics + chart */}
            <div
                style={{
                borderRadius: 16,
                border: "1px solid rgba(255,255,255,0.10)",
                background: "rgba(0,0,0,0.12)",
                overflow: "hidden",
                padding: 12,
                minHeight: 560,
                }}
            >
                <div
                className="chartModalBodyGrid"
                style={{
                    display: "grid",
                    gridTemplateColumns: "360px minmax(0, 1fr)",
                    gap: 14,
                    alignItems: "start",
                }}
                >
                {/* LEFT: metrics (fade in smoothly) */}
                <motion.div
                    initial={{ opacity: 0, y: 6, filter: "blur(6px)" }}
                    animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                    exit={{ opacity: 0, y: 4, filter: "blur(6px)" }}
                    transition={{ duration: 0.45, ease: "easeOut", delay: 0.10 }}
                    style={{
                    padding: 10,
                    borderRadius: 14,
                    // IMPORTANT: no separate “box”, keep it inside same container
                    background: "transparent",
                    border: "none",
                    position: "sticky",
                    top: 14,
                    alignSelf: "start",
                    }}
                >
                    {/* Description */}
                    {subtitle ? (
                    <div style={{ fontSize: 12.5, lineHeight: 1.4, color: "rgba(255,255,255,0.82)" }}>
                        {subtitle}
                    </div>
                    ) : null}

                    {/* Takeaway */}
                    {takeaway ? (
                    <div
                        style={{
                        marginTop: 10,
                        padding: "10px 10px",
                        borderRadius: 12,
                        border: "1px solid rgba(255,255,255,0.10)",
                        background: "rgba(255,255,255,0.04)", // subtle within same container
                        fontSize: 12.5,
                        lineHeight: 1.4,
                        color: "rgba(255,255,255,0.86)",
                        }}
                    >
                        <span className="muted" style={{ fontWeight: 900 }}>
                        Takeaway:
                        </span>{" "}
                        <span style={{ fontWeight: 850 }}>{takeaway}</span>
                    </div>
                    ) : null}
                    {/* Custom sidebar content (optional) */}
                    {sidebar ? (
                    <div style={{ marginTop: 12 }}>
                        {sidebar}
                    </div>
                    ) : null}
                    {/* Metrics table */}
                    {stats?.length ? (
                    <div style={{ marginTop: 12 }}>
                        <div className="muted" style={{ fontWeight: 900, fontSize: 12, marginBottom: 8 }}>
                        Quick metrics
                        </div>

                        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 6 }}>
                        {stats.slice(0, 10).map((s) => (
                            <div
                            key={s.label}
                            style={{
                                display: "flex",
                                justifyContent: "space-between",
                                gap: 10,
                                padding: "8px 10px",
                                borderRadius: 12,
                                border: "1px solid rgba(255,255,255,0.10)",
                                background: "rgba(255,255,255,0.03)",
                            }}
                            >
                            <span className="muted" style={{ fontWeight: 850, fontSize: 12 }}>
                                {s.label}
                            </span>
                            <span style={{ fontWeight: 950, fontVariantNumeric: "tabular-nums", fontSize: 12.5 }}>
                                {s.value}
                            </span>
                            </div>
                        ))}
                        </div>
                    </div>
                    ) : null}

                    {/* Optional LLM panel */}
                    
                </motion.div>

                {/* RIGHT: chart (slide from left + fade) */}
                {/* RIGHT: chart (slide from left + fade) */}
                <motion.div
                key={`chart-${chartKey}`}
                initial={{ x: -420, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: 200, opacity: 0 }}
                transition={{
                    duration: 0.95,
                    delay: 0.18,
                    ease: [0.22, 1, 0.36, 1], // smoother cubic
                }}
                style={{
                    background: "transparent",
                    border: "none",
                    overflow: "hidden",
                    padding: 6,
                    minHeight: 520,
                    willChange: "transform, opacity",
                }}
                >
                <div id={chartKey ? `chart-${chartKey}` : undefined}>
                  {children}
                </div>
                </motion.div>
                </div>
            </div>
            </motion.div>
            </AnimatePresence>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

/** ---------------- main component ---------------- */

export default function AnalyticsPage({
  rows,
  streaming,
}: {
  rows: Row[];
  streaming: boolean;
}) {
  const [priceMetric, setPriceMetric] = useState<"d" | "c" | "tv" | "cpc">("d");
  const [expanded, setExpanded] = useState<ChartId>(null);



  // tooltip trackers (cursor-follow)
  const pieTT = useFollowTooltip();
  const catTT = useFollowTooltip();
  const depthTT = useFollowTooltip();
  const distTT = useFollowTooltip();
  const scatterTT = useFollowTooltip();
  const topTT = useFollowTooltip();

  const llm = useLLMInsights();

  const metricLabel = useMemo(() => {
    if (priceMetric === "d") return "Digital CPM (D-CPM)";
    if (priceMetric === "c") return "Content CPM (C-CPM)";
    if (priceMetric === "tv") return "TV CPM (TV-CPM)";
    return "Cost Per Click (CPC)";
  }, [priceMetric]);

  const axis = {
    tick: { fill: "rgba(255,255,255,0.55)", fontSize: 11 },
    axis: { stroke: "rgba(255,255,255,0.22)" },
    grid: "rgba(255,255,255,0.06)",
  };

  const accent = "var(--accent)";
  const donutColors = [
    `color-mix(in srgb, ${accent} 75%, transparent)`,
    `color-mix(in srgb, ${accent} 55%, transparent)`,
    `color-mix(in srgb, ${accent} 40%, transparent)`,
    `color-mix(in srgb, ${accent} 28%, transparent)`,
    `color-mix(in srgb, ${accent} 18%, transparent)`,
  ];

  const data = useMemo(() => {
    const safe = rows ?? [];

    const prices = safe.map((r) => pickPrice(r, priceMetric)).filter((x): x is number => x != null);
    // ✅ price-ranked rows (for AI outlier insights)
    const pricedRows = safe
      .map((r) => {
        const p = pickPrice(r, priceMetric);
        if (p == null) return null;

        return {
          name: getName(r),
          price: p,
          l1: getL1FromNewSegmentName(r) ?? getL1Category(r) ?? "Uncategorized",
          sources: getSources(r).length,
          rank: pickRank(r),
          uniq: pickUniq(r),
        };
      })
      .filter(Boolean) as any[];

    const topHigh = [...pricedRows].sort((a, b) => b.price - a.price).slice(0, 10);
    const topLow = [...pricedRows].sort((a, b) => a.price - b.price).slice(0, 10);

    // ✅ small category price summary (helps AI compare categories)
    const catPriceSummary = (() => {
      const m = new Map<string, number[]>();
      for (const r of safe) {
        const p = pickPrice(r, priceMetric);
        if (p == null) continue;
        const c = getL1FromNewSegmentName(r) ?? getL1Category(r) ?? "Uncategorized";
        if (!m.has(c)) m.set(c, []);
        m.get(c)!.push(p);
      }

      const arr = Array.from(m.entries()).map(([cat, vals]) => ({
        cat,
        n: vals.length,
        avg: vals.reduce((a, b) => a + b, 0) / vals.length,
        med: median(vals),
        min: Math.min(...vals),
        max: Math.max(...vals),
      }));

      // keep payload small
      return arr.sort((a, b) => b.n - a.n).slice(0, 8);
    })();
    const { min: priceMin, max: priceMax } = minmax(prices);

    const ranks = safe.map((r) => pickRank(r)).filter((x): x is number => x != null);
    const uniqs = safe.map((r) => pickUniq(r)).filter((x): x is number => x != null);

    const { min: rankMin, max: rankMax } = minmax(ranks);
    const { min: uniqMin, max: uniqMax } = minmax(uniqs);

    // scatter points
    const scatter = safe
      .map((r) => {
        const x = pickUniq(r);
        const y = pickRank(r);
        if (x == null || y == null) return null;
        return {
          x,
          y,
          name: getName(r) || "—",
          d: pickPrice(r, "d"),
          c: pickPrice(r, "c"),
          tv: pickPrice(r, "tv"),
          cpc: pickPrice(r, "cpc"),
          sources: getSources(r).length,
        };
      })
      .filter(Boolean) as any[];

    // debug counts for missing fields (helps explain empty scatter)
    const missingRank = safe.filter((r) => pickRank(r) == null).length;
    const missingUniq = safe.filter((r) => pickUniq(r) == null).length;

    // top reused sources
    const counts = new Map<string, number>();
    for (const r of safe) {
      for (const s of getSources(r)) counts.set(s, (counts.get(s) ?? 0) + 1);
    }
    const topReused = Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12)
      .map(([name, uses]) => ({ name, label: cleanSourceLabel(name), uses }));
    const uniqueSourceCount = counts.size;

    // reuse depth (kept as a chart, but NOT your “segment distribution” anymore)
    const sourcesCountDist = safe.map((r) => getSources(r).length);
    const depthCounts = new Map<number, number>();
    for (const d of sourcesCountDist) {
      const key = d >= 5 ? 5 : d;
      depthCounts.set(key, (depthCounts.get(key) ?? 0) + 1);
    }
    const depthBars = [0, 1, 2, 3, 4, 5].map((k) => ({
      depth: k === 5 ? "5+" : String(k),
      count: depthCounts.get(k) ?? 0,
      key: k,
    }));

    // price distributions
    const priceBuckets = bucketize(prices, 18);

    // price donut quartiles
    const priceQuartDonut = (() => {
      if (prices.length === 0) return [];
      const q1 = quantile(prices, 0.25)!;
      const q2 = quantile(prices, 0.5)!;
      const q3 = quantile(prices, 0.75)!;

      const buckets = [
        { name: `≤ ${fmt(q1)}`, lo: -Infinity, hi: q1 },
        { name: `${fmt(q1)}–${fmt(q2)}`, lo: q1, hi: q2 },
        { name: `${fmt(q2)}–${fmt(q3)}`, lo: q2, hi: q3 },
        { name: `≥ ${fmt(q3)}`, lo: q3, hi: Infinity },
      ];

      return buckets
        .map((b) => ({ name: b.name, value: prices.filter((v) => v >= b.lo && v <= b.hi).length }))
        .filter((x) => x.value > 0);
    })();

    // CATEGORY distribution (this is what you asked for)
    const catCounts = new Map<string, number>();

    for (const r of safe) {
    // Prefer: L1 derived from the New Segment Name
    // Fallback: explicit L1/category fields if they exist
    const c = getL1FromNewSegmentName(r) ?? getL1Category(r) ?? "Uncategorized";
    catCounts.set(c, (catCounts.get(c) ?? 0) + 1);
    }

    // top categories + "Other" for readability
    const catSorted = Array.from(catCounts.entries()).sort((a, b) => b[1] - a[1]);
    const topN = 9;
    const topCats = catSorted.slice(0, topN);
    const other = catSorted.slice(topN).reduce((sum, [, v]) => sum + v, 0);

    const categoryDist = [
      ...topCats.map(([name, value]) => ({ name, value })),
      ...(other > 0 ? [{ name: "Other", value: other }] : []),
    ];

    const avgPrice = prices.length ? prices.reduce((a, b) => a + b, 0) / prices.length : null;

    const corr = (() => {
      if (!scatter.length) return null;
      return pearson(
        scatter.map((p) => p.x),
        scatter.map((p) => p.y)
      );
    })();

    return {
      n: safe.length,

      prices,
      priceBuckets,
      priceQuartDonut,
      avgPrice,
      medPrice: prices.length ? median(prices) : null,
      priceMin,
      priceMax,

      ranks,
      uniqs,
      avgRank: ranks.length ? ranks.reduce((a, b) => a + b, 0) / ranks.length : null,
      avgUniq: uniqs.length ? uniqs.reduce((a, b) => a + b, 0) / uniqs.length : null,
      rankMin,
      rankMax,
      uniqMin,
      uniqMax,

      scatter,
      corr,
      missingRank,
      missingUniq,

      depthBars,

      topReused,
      uniqueSourceCount,

      categoryDist,
      categoryTotalUnique: catCounts.size,
      topHigh,
      topLow,
      catPriceSummary,
    };
  }, [rows, priceMetric]);

  const askAI = useCallback(async () => {
  if (!expanded) return;

  // ✅ capture chart as base64 (unique per expanded chart)
  const chartEl = document.getElementById(`chart-${expanded}`) as HTMLElement | null;
  const chartImageB64 = chartEl ? await captureChartBase64(chartEl) : "";

  // Keep payload small: summarize + send a small sample.
  const payload: any = {
    chartId: expanded,
    metric: priceMetric,
    metricLabel,
    totals: {
      rows: data.n,
      nonNullPrice: data.prices.length,
      pointsScatter: data.scatter.length,
    },
    stats: {
      avgPrice: data.avgPrice,
      medPrice: data.medPrice,
      minPrice: data.priceMin,
      maxPrice: data.priceMax,
      avgRank: data.avgRank,
      avgUniq: data.avgUniq,
      corr: data.corr,
    },
    sample: {
      topCategories: data.categoryDist.slice(0, 10),
      topReused: data.topReused.slice(0, 10),
      topHigh: data.topHigh,
      topLow: data.topLow,
      catPriceSummary: data.catPriceSummary,
      priceBuckets: data.priceBuckets.slice(0, 10),
      scatter: data.scatter.slice(0, 15),
    },

    // ✅ add the visual
    chartImageB64,
  };

  llm.run(payload);
}, [expanded, priceMetric, metricLabel, data, llm]);

  const modal = useMemo(() => {
    const base = {
      open: expanded !== null,
      onClose: () => setExpanded(null),
      title: "",
      subtitle: "",
      takeaway: "",
      sidebar: null as React.ReactNode,
      stats: [] as { label: string; value: string }[],
      content: null as React.ReactNode,
    };

    if (!expanded) return base;

    if (expanded === "priceBuckets") {
      const n = data.prices.length;
      const q1 = n ? quantile(data.prices, 0.25) : null;
      const q3 = n ? quantile(data.prices, 0.75) : null;

      base.title = "Price tiers (quartiles)";
      base.subtitle = "How prices split into 4 buckets. Useful for spotting whether pricing is tight or spread out.";
      base.takeaway =
        n === 0
          ? `No ${metricLabel} values yet.`
          : `Middle 50% is roughly ${fmt(q1)}–${fmt(q3)}. Outliers outside this band are worth auditing.`;
      base.stats = [
        { label: "Rows", value: String(data.n) },
        { label: `Non-null prices`, value: String(n) },
        { label: "Avg price", value: fmt(data.avgPrice) },
        { label: "Median price", value: fmt(data.medPrice) },
        { label: "Min price", value: fmt(data.priceMin) },
        { label: "Max price", value: fmt(data.priceMax) },
      ];
      base.sidebar = (
        <div
            style={{
            padding: "12px",
            borderRadius: 14,
            border: "1px solid rgba(255,255,255,0.10)",
            background: "rgba(255,255,255,0.04)",
            display: "grid",
            gap: 10,
            }}
        >
            <div style={{ fontWeight: 900, fontSize: 12, opacity: 0.7 }}>
            Price structure insight
            </div>

            <div style={{ fontSize: 12.5, lineHeight: 1.5 }}>
            <b>Spread:</b>{" "}
            {data.priceMin != null && data.priceMax != null
                ? fmt((data.priceMax ?? 0) - (data.priceMin ?? 0))
                : "—"}
            </div>

            <div style={{ fontSize: 12.5 }}>
            <b>Middle 50% band:</b>{" "}
            {data.prices.length
                ? `${fmt(quantile(data.prices, 0.25))} – ${fmt(
                    quantile(data.prices, 0.75)
                )}`
                : "—"}
            </div>

            <div style={{ fontSize: 12.5 }}>
            <b>Distribution shape:</b>{" "}
            {data.priceQuartDonut.length === 4
                ? "Evenly distributed"
                : "Skewed toward one side"}
            </div>
        </div>
        );
      base.content = (
        <div style={{ height: 520, padding: 6 }}>
          {data.priceQuartDonut.length === 0 ? (
            <div className="muted" style={{ padding: 14 }}>
              No prices available yet.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%" debounce={50}>
              <PieChart onMouseMove={pieTT.onMove} onMouseLeave={pieTT.onLeave}>
                <CursorOnlyTooltip
                  pt={pieTT.pt}
                  content={({ active, payload, label }: any) =>
                    pieTT.pt ? <CursorTooltip active={active} payload={payload} label={label} /> : null
                  }
                />

                <Pie
                  data={data.priceQuartDonut}
                  dataKey="value"
                  nameKey="name"
                  innerRadius="62%"
                  outerRadius="88%"
                  paddingAngle={2}
                  label={pieOuterLabel}
                  labelLine={false}
                  minAngle={6}
                  isAnimationActive={false}   // ✅ important: stops re-anim/jitter on hover
                >
                  {data.priceQuartDonut.map((_, i) => (
                    <Cell
                      key={`cell-${i}`}
                      fill={donutColors[i % donutColors.length]}
                      stroke="rgba(255,255,255,0.18)"
                    />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      );
      return base;
    }

    if (expanded === "priceDist") {
      const n = data.prices.length;
      const q10 = n ? quantile(data.prices, 0.1) : null;
      const q90 = n ? quantile(data.prices, 0.9) : null;

      base.title = "Price distribution (bucketed)";
      base.subtitle = "Counts of segments across price buckets (helps reveal spikes and long tails).";
      base.takeaway =
        n === 0
          ? `No ${metricLabel} values yet.`
          : `Typical range (10th–90th percentile) is ~${fmt(q10)}–${fmt(q90)}. Look at tails beyond this range for outliers.`;
      base.stats = [
        { label: "Rows", value: String(data.n) },
        { label: "Non-null prices", value: String(n) },
        { label: "Avg price", value: fmt(data.avgPrice) },
        { label: "Median price", value: fmt(data.medPrice) },
        { label: "Min price", value: fmt(data.priceMin) },
        { label: "Max price", value: fmt(data.priceMax) },
      ];

      base.content = (
        <div style={{ height: 520, padding: 6 }}>
          {data.priceBuckets.length === 0 ? (
            <div className="muted" style={{ padding: 14 }}>
              No prices available yet.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%" debounce={60}>
              <AreaChart
                data={data.priceBuckets}
                onMouseMove={distTT.onMove}
                onMouseLeave={distTT.onLeave}
                margin={{ left: 10, right: 14, top: 8, bottom: 10 }}
              >
                <CartesianGrid stroke={axis.grid} />
                <XAxis dataKey="x" tickFormatter={(v) => fmt(v)} stroke={axis.axis.stroke} tick={axis.tick as any} />
                <YAxis stroke={axis.axis.stroke} tick={axis.tick as any} />
                <Tooltip
                  content={({ active, payload }: any) => {
                    if (!distTT.pt) return null;
                    if (!active || !payload?.length) return null;
                    const p = payload[0]?.payload;
                    return (
                      <CursorTooltip
                        active={active}
                        payload={[
                          { name: "Price range", value: `${fmt(p?.lo ?? null)}–${fmt(p?.hi ?? null)}` },
                          { name: "Count", value: p?.y ?? 0 },
                        ]}
                        label="Bucket"
                      />
                    );
                  }}
                  position={distTT.pt ? { x: distTT.pt.x + 14, y: distTT.pt.y - 12 } : { x: -9999, y: -9999 }}
                  allowEscapeViewBox={{ x: true, y: true }}
                  wrapperStyle={{ outline: "none", pointerEvents: "none" }}
                />
                <Area
                  type="monotone"
                  dataKey="y"
                  name="count"
                  stroke={`color-mix(in srgb, ${accent} 70%, rgba(255,255,255,0.35))`}
                  fill={`color-mix(in srgb, ${accent} 22%, transparent)`}
                  isAnimationActive
                  animationDuration={820}
                  animationBegin={40}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      );
      return base;
    }

    if (expanded === "categoryDist") {
      const top = data.categoryDist[0];

      base.title = "Segment distribution by category";
      base.subtitle = "How many segments fall into each taxonomy/category (B2B, Functional, etc.).";
      base.takeaway =
        data.n === 0
          ? "No rows loaded."
          : top
          ? `Largest category is "${top.name}" with ${top.value} segments (${pct(top.value, data.n)}%).`
          : "No categories detected (all rows are Uncategorized).";
      base.stats = [
        { label: "Rows", value: String(data.n) },
        { label: "Unique categories", value: String(data.categoryTotalUnique) },
        { label: "Top category", value: String(top?.name ?? "—") },
        { label: "Top count", value: String(top?.value ?? 0) },
        { label: "Uncategorized", value: String(data.categoryDist.find((x) => x.name === "Uncategorized")?.value ?? 0) },
        { label: "Other", value: String(data.categoryDist.find((x) => x.name === "Other")?.value ?? 0) },
      ];

      base.sidebar = (
        <div
            style={{
            padding: "12px",
            borderRadius: 14,
            border: "1px solid rgba(255,255,255,0.10)",
            background: "rgba(255,255,255,0.04)",
            display: "grid",
            gap: 10,
            }}
        >
            <div style={{ fontWeight: 900, fontSize: 12, opacity: 0.7 }}>
            Category structure insight
            </div>

            <div style={{ fontSize: 12.5 }}>
            <b>Largest share:</b>{" "}
            {top ? `${top.name} (${pct(top.value, data.n)}%)` : "—"}
            </div>

            <div style={{ fontSize: 12.5 }}>
            <b>Concentration:</b>{" "}
            {top && pct(top.value, data.n) > 40
                ? "Highly concentrated"
                : top && pct(top.value, data.n) > 25
                ? "Moderately concentrated"
                : "Diversified"}
            </div>

            <div style={{ fontSize: 12.5 }}>
            <b>Fragmentation level:</b>{" "}
            {data.categoryTotalUnique > 8
                ? "High"
                : data.categoryTotalUnique > 4
                ? "Medium"
                : "Low"}
            </div>
        </div>
        );

      base.content = (
        <div style={{ height: 520, padding: 6 }}>
          {data.categoryDist.length === 0 ? (
            <div className="muted" style={{ padding: 14 }}>
              No category data available. Add a Category/Segment Type column to the export.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%" debounce={50}>
              <PieChart onMouseMove={catTT.onMove} onMouseLeave={catTT.onLeave}>
                <CursorOnlyTooltip
                  pt={catTT.pt}
                  content={({ active, payload, label }: any) =>
                    catTT.pt ? <CursorTooltip active={active} payload={payload} label={label} /> : null
                  }
                />

                <Pie
                  data={data.categoryDist}
                  dataKey="value"
                  nameKey="name"
                  outerRadius="88%"
                  paddingAngle={1}
                  label={pieOuterLabel}
                  labelLine={false}
                  minAngle={6}
                  isAnimationActive={false}   // ✅ same fix
                >
                  {data.categoryDist.map((_, i) => (
                    <Cell
                      key={`catcell-${i}`}
                      fill={donutColors[i % donutColors.length]}
                      stroke="rgba(255,255,255,0.18)"
                    />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      );
      return base;
    }

    if (expanded === "reuseDepth") {
      base.title = "Reuse depth (sources per derived segment)";
      base.subtitle = "How many source segments were used to build each segment (0,1,2,3,4,5+).";
      base.takeaway = "Higher values indicate more composition (re-use) from existing sources.";
      base.stats = [
        { label: "Rows", value: String(data.n) },
        { label: "0 sources", value: String(data.depthBars.find((x) => x.depth === "0")?.count ?? 0) },
        { label: "1 source", value: String(data.depthBars.find((x) => x.depth === "1")?.count ?? 0) },
        { label: "2 sources", value: String(data.depthBars.find((x) => x.depth === "2")?.count ?? 0) },
        { label: "3 sources", value: String(data.depthBars.find((x) => x.depth === "3")?.count ?? 0) },
        { label: "5+ sources", value: String(data.depthBars.find((x) => x.depth === "5+")?.count ?? 0) },
      ];

    const zero = data.depthBars.find(x => x.depth === "0")?.count ?? 0;
    const fivePlus = data.depthBars.find(x => x.depth === "5+")?.count ?? 0;

    base.sidebar = (
    <div
        style={{
        padding: "12px",
        borderRadius: 14,
        border: "1px solid rgba(255,255,255,0.10)",
        background: "rgba(255,255,255,0.04)",
        display: "grid",
        gap: 10,
        }}
    >
        <div style={{ fontWeight: 900, fontSize: 12, opacity: 0.7 }}>
        Composition insight
        </div>

        <div style={{ fontSize: 12.5 }}>
        <b>No-source segments:</b> {pct(zero, data.n)}%
        </div>

        <div style={{ fontSize: 12.5 }}>
        <b>Heavy compositions (5+):</b> {pct(fivePlus, data.n)}%
        </div>

        <div style={{ fontSize: 12.5 }}>
        <b>Overall reuse level:</b>{" "}
        {fivePlus > zero ? "High reuse ecosystem" : "Light reuse"}
        </div>
    </div>
    );
      base.content = (
        <div style={{ height: 520, padding: 6 }}>
          <ResponsiveContainer width="100%" height="100%" debounce={60}>
            <BarChart
              data={data.depthBars}
              onMouseMove={depthTT.onMove}
              onMouseLeave={depthTT.onLeave}
            >
              <CartesianGrid stroke={axis.grid} />
              <XAxis dataKey="depth" stroke={axis.axis.stroke} tick={axis.tick as any} />
              <YAxis stroke={axis.axis.stroke} tick={axis.tick as any} />

              <CursorOnlyTooltip
                pt={depthTT.pt}
                content={({ active, payload, label }: any) =>
                  depthTT.pt ? <CursorTooltip active={active} payload={payload} label={label} /> : null
                }
              />

              <Bar
                dataKey="count"
                name="segments"
                fill={`color-mix(in srgb, ${accent} 28%, transparent)`}
                stroke={`color-mix(in srgb, ${accent} 55%, rgba(255,255,255,0.22))`}
                isAnimationActive
                animationDuration={720}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      );
      return base;
    }

    if (expanded === "scatter") {
      const corr = data.corr;
      const corrLabel =
        corr == null
          ? "—"
          : corr > 0.6
          ? "strong +"
          : corr > 0.25
          ? "moderate +"
          : corr > -0.25
          ? "weak"
          : corr > -0.6
          ? "moderate −"
          : "strong −";

      base.title = "Rank vs Uniqueness (segment map)";
      base.subtitle = "Each point is a segment. Hover points to see the segment + its prices.";
      base.takeaway =
        data.scatter.length < 2
          ? `Empty because rank/uniqueness are missing. Missing rank: ${data.missingRank}/${data.n}. Missing uniqueness: ${data.missingUniq}/${data.n}.`
          : `Correlation looks ${corrLabel} (r=${fmt(corr)}). Look for top-right clusters (high rank + high uniqueness).`;
      base.stats = [
        { label: "Rows", value: String(data.n) },
        { label: "Points (rank+uniq)", value: String(data.scatter.length) },
        { label: "Avg rank", value: fmt(data.avgRank) },
        { label: "Avg uniq", value: fmt(data.avgUniq) },
        { label: "Rank range", value: `${fmt(data.rankMin)} – ${fmt(data.rankMax)}` },
        { label: "Uniq range", value: `${fmt(data.uniqMin)} – ${fmt(data.uniqMax)}` },
      ];

      base.content = (
        <div style={{ height: 560, padding: 6 }}>
          {data.scatter.length === 0 ? (
            <div className="muted" style={{ padding: 14 }}>
              No points to plot yet.
              <div style={{ marginTop: 8, opacity: 0.85 }}>
                We couldn’t find both fields on the same rows. Common headers include:
                <div style={{ marginTop: 6 }}>
                  <code>Rank Score</code>, <code>rank_score</code>, <code>Rank</code> and{" "}
                  <code>Uniqueness Score</code>, <code>uniqueness_score</code>, <code>Uniq</code>.
                </div>
              </div>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%" debounce={60}>
              <ScatterChart
                onMouseMove={scatterTT.onMove}
                onMouseLeave={scatterTT.onLeave}
                margin={{ left: 10, right: 14, top: 8, bottom: 10 }}
              >
                <CartesianGrid stroke={axis.grid} />
                <XAxis type="number" dataKey="x" name="uniqueness" stroke={axis.axis.stroke} tick={axis.tick as any} />
                <YAxis type="number" dataKey="y" name="rank" stroke={axis.axis.stroke} tick={axis.tick as any} />
                <Tooltip
                  content={({ active, payload }: any) => {
                    if (!scatterTT.pt) return null;
                    if (!active || !payload?.length) return null;
                    const p = payload[0]?.payload;
                    return (
                      <CursorTooltip
                        active={active}
                        label={p?.name ?? "Segment"}
                        payload={[
                          { name: "Sources used", value: p?.sources ?? 0 },
                          ...(p?.d != null ? [{ name: "D-CPM", value: p.d }] : []),
                          ...(p?.c != null ? [{ name: "C-CPM", value: p.c }] : []),
                          ...(p?.tv != null ? [{ name: "TV-CPM", value: p.tv }] : []),
                          ...(p?.cpc != null ? [{ name: "CPC", value: p.cpc }] : []),
                          { name: "Uniqueness", value: p?.x ?? null },
                          { name: "Rank", value: p?.y ?? null },
                        ]}
                      />
                    );
                  }}
                  position={scatterTT.pt ? { x: scatterTT.pt.x + 14, y: scatterTT.pt.y - 12 } : { x: -9999, y: -9999 }}
                  allowEscapeViewBox={{ x: true, y: true }}
                  wrapperStyle={{ outline: "none", pointerEvents: "none" }}
                />
                <Scatter
                  name="segments"
                  data={data.scatter}
                  fill={`color-mix(in srgb, ${accent} 26%, rgba(255,255,255,0.10))`}
                  isAnimationActive
                  animationDuration={780}
                />
              </ScatterChart>
            </ResponsiveContainer>
          )}
        </div>
      );
      return base;
    }

    if (expanded === "topReused") {
      const top1 = data.topReused[0]?.name;
      const top1Uses = data.topReused[0]?.uses ?? 0;

      base.title = "Most reused source segments";
      base.subtitle = "Which source segments are most frequently used as building blocks.";
      base.takeaway =
        data.uniqueSourceCount === 0
          ? "No source segments detected."
          : top1
          ? `You have ${data.uniqueSourceCount} unique source segments. The most reused appears ${top1Uses} times.`
          : `You have ${data.uniqueSourceCount} unique source segments.`;
      base.stats = [
        { label: "Rows", value: String(data.n) },
        { label: "Unique source segments", value: String(data.uniqueSourceCount) },
        { label: "Top list size", value: String(data.topReused.length) },
      ];

      base.content = (
        <div style={{ height: 600, padding: 6 }}>
          <ResponsiveContainer width="100%" height="100%" debounce={60}>
            <BarChart
              data={data.topReused}
              layout="vertical"
              onMouseMove={topTT.onMove}
              onMouseLeave={topTT.onLeave}
              margin={{ left: 16, right: 18, top: 8, bottom: 10 }}
            >
              <CartesianGrid stroke={axis.grid} />
              <XAxis type="number" stroke={axis.axis.stroke} tick={axis.tick as any} />
              <YAxis type="category" dataKey="label" width={300} stroke={axis.axis.stroke} tick={axis.tick as any} />
              <Tooltip
                content={({ active, payload }: any) =>
                  topTT.pt ? <CursorTooltip active={active} payload={payload} label="Source segment" /> : null
                }
                position={topTT.pt ? { x: topTT.pt.x + 14, y: topTT.pt.y - 12 } : { x: -9999, y: -9999 }}
                allowEscapeViewBox={{ x: true, y: true }}
                wrapperStyle={{ outline: "none", pointerEvents: "none" }}
              />
              <Bar
                dataKey="uses"
                name="uses"
                fill={`color-mix(in srgb, ${accent} 18%, transparent)`}
                stroke={`color-mix(in srgb, ${accent} 55%, rgba(255,255,255,0.22))`}
                isAnimationActive
                animationDuration={780}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      );
      return base;
    }

    return base;
  }, [
    expanded,
    data,
    metricLabel,
    donutColors,
    accent,
    axis,
    pieTT,
    catTT,
    depthTT,
    distTT,
    scatterTT,
    topTT,
  ]);

  // clear AI text when switching charts (optional; makes UX cleaner)
  useEffect(() => {
    llm.clear();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expanded]);

  return (
    <div className="pageFade">
      <ChartModal
        open={modal.open}
        onClose={modal.onClose}
        title={modal.title}
        subtitle={modal.subtitle}
        takeaway={modal.takeaway}
        sidebar={(modal as any).sidebar}
        stats={modal.stats}
        llm={llm.state}
        onAskAI={askAI}
        chartKey={expanded}
      >
        {modal.content}
      </ChartModal>

      {/* HEADER STRIP */}
      <section className="filtersStrip fadeInUp">
        <div className="filtersStripHeader">
          <div className="cardTitle">Analytics</div>
          <div className="cardHint">
            Showing <b>{data.n}</b> rows
          </div>
        </div>

        <div className="filtersStripBody">
          <div className="field" style={{ minWidth: 320 }}>
            <span>Price metric</span>
            <MetricTabs value={priceMetric} onChange={setPriceMetric} disabled={streaming} />
          </div>

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

          <div className="field" style={{ minWidth: 320 }}>
            <span>Key metrics</span>
            <div style={{ marginTop: 10, display: "grid", gap: 6 }}>
              <StatRow label={`Avg price (${metricLabel})`} value={fmt(data.avgPrice)} />
              <StatRow label={`Median price (${metricLabel})`} value={fmt(data.medPrice)} />
              <StatRow label="Avg rank" value={fmt(data.avgRank)} />
              <StatRow label="Avg uniqueness" value={fmt(data.avgUniq)} />
            </div>
          </div>
        </div>
      </section>

      {/* CHART GRID */}
      <div
        className="fadeInUp"
        style={{
          display: "grid",
          gridTemplateColumns: "1.2fr 1fr",
          gap: 14,
        }}
      >
        {/* Price tiers donut */}
        <CardEnter k={`donut-${priceMetric}`} delay={0.02} onClick={() => setExpanded("priceBuckets")}>
          <div className="cardHeader">
            <div className="cardTitle">Price tiers (quartiles)</div>
            <div className="muted">How prices split into four buckets</div>
          </div>

          <div className="miniStats">
            <span className="miniStat"><span className="miniStatKey">priced rows</span> {data.prices.length}</span>
            <span className="miniStat"><span className="miniStatKey">min</span> {fmt(data.priceMin)}</span>
            <span className="miniStat"><span className="miniStatKey">max</span> {fmt(data.priceMax)}</span>
          </div>

          <div style={{ height: 260, padding: 10 }}>
            {data.priceQuartDonut.length === 0 ? (
              <div className="muted" style={{ padding: 14 }}>No prices available yet.</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%" debounce={50}>
                <PieChart>
                  <Tooltip content={<GlassTooltip />} wrapperStyle={{ outline: "none", pointerEvents: "none" }} />
                  <Pie
                    data={data.priceQuartDonut}
                    dataKey="value"
                    nameKey="name"
                    innerRadius="62%"
                    outerRadius="88%"
                    paddingAngle={2}
                    label={pieOuterLabel}
                    labelLine={false}
                    minAngle={6}
                    isAnimationActive
                    animationBegin={80}
                    animationDuration={850}
                    >
                    {data.priceQuartDonut.map((_, i) => (
                      <Cell
                        key={`cell-${i}`}
                        fill={donutColors[i % donutColors.length]}
                        stroke="rgba(255,255,255,0.18)"
                      />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </CardEnter>

        {/* Segment distribution by category */}
        <CardEnter k={`catdist-${priceMetric}`} delay={0.045} onClick={() => setExpanded("categoryDist")}>
          <div className="cardHeader">
            <div className="cardTitle">Segment distribution (by category)</div>
            <div className="muted">B2B vs Functional vs …</div>
          </div>

          <div className="miniStats">
            <span className="miniStat"><span className="miniStatKey">categories</span> {data.categoryTotalUnique}</span>
            <span className="miniStat"><span className="miniStatKey">top</span> {data.categoryDist[0]?.name ?? "—"}</span>
            <span className="miniStat"><span className="miniStatKey">top %</span> {data.categoryDist[0] ? `${pct(data.categoryDist[0].value, data.n)}%` : "—"}</span>
          </div>

          <div style={{ height: 260, padding: 10 }}>
            {data.categoryDist.length === 0 ? (
              <div className="muted" style={{ padding: 14 }}>
                No category data found. Add a Category/Segment Type column.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%" debounce={50}>
                <PieChart>
                  <Tooltip content={<GlassTooltip />} wrapperStyle={{ outline: "none", pointerEvents: "none" }} />
                  <Pie
                    data={data.categoryDist}
                    dataKey="value"
                    nameKey="name"
                    outerRadius="88%"
                    paddingAngle={1}
                    label={pieOuterLabel}
                    labelLine={false}
                    minAngle={6}
                    isAnimationActive
                    animationBegin={40}
                    animationDuration={700}
                    >
                    {data.categoryDist.map((_, i) => (
                        <Cell
                        key={`catcell-${i}`}
                        fill={donutColors[i % donutColors.length]}
                        stroke="rgba(255,255,255,0.18)"
                        />
                    ))}
                    </Pie>
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </CardEnter>

        {/* Reuse depth bar (kept, but now clearly named) */}
        <CardEnter k={`depth-${priceMetric}`} delay={0.07} onClick={() => setExpanded("reuseDepth")}>
          <div className="cardHeader">
            <div className="cardTitle">Reuse depth (sources per segment)</div>
            <div className="muted">How many source segments build each segment</div>
          </div>

          <div style={{ height: 260, padding: 10 }}>
            <ResponsiveContainer width="100%" height="100%" debounce={50}>
              <BarChart data={data.depthBars}>
                <CartesianGrid stroke={axis.grid} />
                <XAxis dataKey="depth" stroke={axis.axis.stroke} tick={axis.tick as any} />
                <YAxis stroke={axis.axis.stroke} tick={axis.tick as any} />
                <Tooltip content={<GlassTooltip />} wrapperStyle={{ outline: "none", pointerEvents: "none" }} />
                <Bar
                  dataKey="count"
                  name="segments"
                  fill={`color-mix(in srgb, ${accent} 28%, transparent)`}
                  stroke={`color-mix(in srgb, ${accent} 55%, rgba(255,255,255,0.22))`}
                  isAnimationActive
                  animationDuration={720}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardEnter>

        {/* Price distribution */}
        <CardEnter k={`dist-${priceMetric}`} delay={0.095} onClick={() => setExpanded("priceDist")}>
          <div className="cardHeader">
            <div className="cardTitle">Price distribution (bucketed)</div>
            <div className="muted">Where prices cluster and where outliers live</div>
          </div>

          <div style={{ height: 260, padding: 10 }}>
            {data.priceBuckets.length === 0 ? (
              <div className="muted" style={{ padding: 14 }}>No prices available yet.</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%" debounce={50}>
                <AreaChart data={data.priceBuckets}>
                  <CartesianGrid stroke={axis.grid} />
                  <XAxis dataKey="x" tickFormatter={(v) => fmt(v)} stroke={axis.axis.stroke} tick={axis.tick as any} />
                  <YAxis stroke={axis.axis.stroke} tick={axis.tick as any} />
                  <Tooltip
                    content={({ active, payload }: any) => {
                      if (!active || !payload?.length) return null;
                      const p = payload[0]?.payload;
                      return (
                        <GlassTooltip
                          active={active}
                          label="Bucket"
                          payload={[
                            { name: "Price range", value: `${fmt(p?.lo ?? null)}–${fmt(p?.hi ?? null)}` },
                            { name: "Count", value: p?.y ?? 0 },
                          ]}
                        />
                      );
                    }}
                    wrapperStyle={{ outline: "none", pointerEvents: "none" }}
                  />
                  <Area
                    type="monotone"
                    dataKey="y"
                    name="count"
                    stroke={`color-mix(in srgb, ${accent} 70%, rgba(255,255,255,0.35))`}
                    fill={`color-mix(in srgb, ${accent} 22%, transparent)`}
                    isAnimationActive
                    animationDuration={820}
                    animationBegin={60}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </CardEnter>

        {/* Scatter */}
        <CardEnter k={`scatter-${priceMetric}`} delay={0.12} onClick={() => setExpanded("scatter")}>
          <div className="cardHeader">
            <div className="cardTitle">Rank vs Uniqueness (segment map)</div>
            <div className="muted">Find high-rank + high-uniqueness segments</div>
          </div>

          <div className="miniStats">
            <span className="miniStat"><span className="miniStatKey">points</span> {data.scatter.length}</span>
            <span className="miniStat"><span className="miniStatKey">corr</span> {data.corr == null ? "—" : fmt(data.corr)}</span>
            <span className="miniStat"><span className="miniStatKey">missing rank</span> {data.missingRank}</span>
          </div>

          <div style={{ height: 320, padding: 10 }}>
            {data.scatter.length === 0 ? (
              <div className="muted" style={{ padding: 14 }}>
                Empty — rank/uniqueness not found on rows.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%" debounce={50}>
                <ScatterChart>
                  <CartesianGrid stroke={axis.grid} />
                  <XAxis type="number" dataKey="x" name="uniqueness" stroke={axis.axis.stroke} tick={axis.tick as any} />
                  <YAxis type="number" dataKey="y" name="rank" stroke={axis.axis.stroke} tick={axis.tick as any} />
                  <Tooltip content={<GlassTooltip />} wrapperStyle={{ outline: "none", pointerEvents: "none" }} />
                  <Scatter
                    name="segments"
                    data={data.scatter}
                    fill={`color-mix(in srgb, ${accent} 22%, rgba(255,255,255,0.10))`}
                    isAnimationActive
                    animationDuration={780}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </div>
        </CardEnter>

        {/* Top reused */}
        <CardEnter k={`top-${priceMetric}`} delay={0.145} onClick={() => setExpanded("topReused")}>
          <div className="cardHeader">
            <div className="cardTitle">Most reused source segments</div>
            <div className="muted">Which sources appear most often in compositions</div>
          </div>

          <div style={{ height: 340, padding: 10 }}>
            <ResponsiveContainer width="100%" height="100%" debounce={50}>
              <BarChart data={data.topReused} layout="vertical" margin={{ left: 10, right: 14, top: 6, bottom: 6 }}>
                <CartesianGrid stroke={axis.grid} />
                <XAxis type="number" stroke={axis.axis.stroke} tick={axis.tick as any} />
                <YAxis type="category" dataKey="label" width={260} stroke={axis.axis.stroke} tick={axis.tick as any} />
                <Tooltip content={<GlassTooltip />} wrapperStyle={{ outline: "none", pointerEvents: "none" }} />
                <Bar
                  dataKey="uses"
                  name="uses"
                  fill={`color-mix(in srgb, ${accent} 18%, transparent)`}
                  stroke={`color-mix(in srgb, ${accent} 55%, rgba(255,255,255,0.22))`}
                  isAnimationActive
                  animationDuration={780}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardEnter>
      </div>

      <style>{`
        /* Segmented tabs */
        .segTabs{
          margin-top: 10px;
          display: inline-flex;
          gap: 6px;
          padding: 6px;
          border-radius: 14px;
          border: 1px solid rgba(255,255,255,0.10);
          background: rgba(255,255,255,0.05);
          box-shadow: 0 0 0 1px rgba(255,255,255,0.03) inset;
        }
        .segTabsDisabled{ opacity: 0.65; }
        .segTab{
          border: 1px solid rgba(255,255,255,0.10);
          background: rgba(0,0,0,0.10);
          color: rgba(255,255,255,0.78);
          font-weight: 900;
          font-size: 12px;
          letter-spacing: 0.2px;
          padding: 8px 10px;
          border-radius: 12px;
          cursor: pointer;
          transition: background 140ms ease, border-color 140ms ease, color 140ms ease;
        }
        .segTab:hover{ color: rgba(255,255,255,0.92); }
        .segTabActive{
          background: color-mix(in srgb, var(--accent) 18%, rgba(255,255,255,0.06));
          border-color: color-mix(in srgb, var(--accent) 45%, rgba(255,255,255,0.12));
          color: rgba(255,255,255,0.92);
          box-shadow: 0 10px 26px rgba(0,0,0,0.18);
        }
        .segTab:disabled{ cursor: default; }

        /* clickable cards */
        .cardClickable{
          cursor: zoom-in;
        }
        /* IMPORTANT: remove hover lift/highlight entirely (your request #5) */
        .cardClickable:hover{
          transform: none !important;
          border-color: inherit !important;
        }

        /* mini stats */
        .miniStats{
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
          padding: 0 14px;
          margin-top: -2px;
          margin-bottom: 6px;
        }
        .miniStat{
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.05);
          font-weight: 900;
          font-size: 12px;
          white-space: nowrap;
          font-variant-numeric: tabular-nums;
          color: rgba(255,255,255,0.88);
        }
        .miniStatKey{
          opacity: 0.72;
          font-weight: 850;
        }

        /* Recharts tick styling */
        .recharts-cartesian-axis-tick-value {
          font-size: 11px !important;
          fill: rgba(255,255,255,0.55) !important;
        }
      `}</style>
    </div>
  );
}