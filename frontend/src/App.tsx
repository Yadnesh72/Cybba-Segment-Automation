import React, { useEffect, useMemo, useRef, useState } from "react";
import { ThemeProvider } from "./Theme";import ReactDOM from "react-dom";
import {
  downloadCsv,
  runPipelineStream,
  RunSummary,
  getRun,
  getLastRun,
  resetLastRun,
  UiSettings
} from "./api";
import DownloadButton from "../components/DownloadButton";
import SegmentsTable from "../components/SegmentsTable";
import SideMenu from "../components/SideMenu";
import cybbaLogo from "./assets/cybba_logo.png";
import ComparisonPage from "../components/ComparisonPage";
import AnalyticsPage from "../components/AnalyticsPage";
import "../styles.css";
import CybbaSegmentsPage from "../components/CybbaSegmentsPage";
import SuggestionsPage from "../components/SuggestionsPage";
//import ConfirmResetModal from "../components/ConfirmResetModal";

type Mode = "final" | "validated";


const STAMPS = [100, 200, 300, 400, 500];

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function toNum(v: any): number | null {
  if (v === null || v === undefined || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

function avg(nums: number[]) {
  if (!nums.length) return null;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function fmt3(v: number | null) {
  return v == null ? "—" : v.toFixed(3);
}

// Stable dedupe key
function rowKey(r: Record<string, any>) {
  return String(
    r?.["New Segment Name"] ??
      r?.["Proposed New Segment Name"] ??
      r?.["Segment Name"] ??
      r?.id ??
      ""
  ).trim();
}

/** ---------------- Tooltip (cursor-follow, portal, no-flicker) ---------------- */
type TipState = {
  title: string;
  body: string;
  x: number;
  y: number;
} | null;

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

    // Use rAF so rapid mousemove doesn't cause layout thrash/flicker
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const pad = 12;
      const gap = 14;

      const vw = window.innerWidth;
      const vh = window.innerHeight;

      const width = 340; // stable width gives stable positioning
      const estH = 92; // good enough for flip logic (title + 2 lines)

      let left = tip.x + gap;
      let top = tip.y + gap;

      // clamp right
      left = clamp(left, pad, vw - width - pad);

      // flip above cursor if near bottom
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
      style={{
        left: pos.left,
        top: pos.top,
      }}
      role="tooltip"
      aria-hidden={!tip}
    >
      <div className="kpiTipTitle">{tip?.title ?? ""}</div>
      <div className="kpiTipBody">{tip?.body ?? ""}</div>
    </div>
  );

  return ReactDOM.createPortal(node, document.body);
}

/** ---------- KPI strip ---------- */
function KpiStrip({
  runId,
  rows,
  summary,
  streaming,
}: {
  runId: string | null;
  rows: Record<string, any>[];
  summary: RunSummary | null;
  streaming: boolean;
}) {
  const rankAvg =
    toNum(summary?.rank_stats?.mean) ??
    avg(rows.map((r) => toNum(r.rank_score)).filter((x): x is number => x != null));

  const uniqAvg =
    toNum(summary?.uniqueness_stats?.mean) ??
    avg(rows.map((r) => toNum(r.uniqueness_score)).filter((x): x is number => x != null));

  const compAvg = avg(
    rows.map((r) => toNum(r["Composition Similarity"])).filter((x): x is number => x != null)
  );

  const closestAvg = avg(
    rows.map((r) => toNum(r["Closest Cybba Similarity"])).filter((x): x is number => x != null)
  );

  const tiles = [
    {
      label: "Rows",
      value: String(rows.length),
      hint: runId ? "current run" : "—",
      tipTitle: "Rows",
      tipBody: "Number of priced segments currently displayed in the KPI strip (based on the rows passed in).",
    },
    {
      label: "Rank avg",
      value: fmt3(rankAvg),
      hint: "quality",
      tipTitle: "Rank average",
      tipBody: "Average rank score across displayed rows. Higher usually indicates better overall segment quality.",
    },
    {
      label: "Uniq avg",
      value: fmt3(uniqAvg),
      hint: "novelty",
      tipTitle: "Uniqueness average",
      tipBody: "Average uniqueness score. Higher means segments are more distinct / less redundant.",
    },
    {
      label: "Comp avg",
      value: fmt3(compAvg),
      hint: "composition",
      tipTitle: "Composition similarity",
      tipBody: "Average similarity to the intended composition features. Higher means closer to the target structure.",
    },
    {
      label: "Closest avg",
      value: fmt3(closestAvg),
      hint: "closest",
      tipTitle: "Closest similarity",
      tipBody: "Average similarity to the nearest existing Cybba segment. Lower can imply more net-new segments.",
    },
  ];

  
  // tooltip state + delayed show for polish
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
      setTip({
        title,
        body,
        x: lastMouseRef.current.x,
        y: lastMouseRef.current.y,
      });
    }, 120); // ✅ show delay
  };

  const hide = () => {
    clearShowTimer();
    setTip(null); // fade-out handled by CSS transition
  };

  useEffect(() => {
    return () => {
      clearShowTimer();
    };
  }, []);

  return (
    <>
      <TooltipPortal tip={tip} />

      <section className="metricsStrip fadeInUp">
        {tiles.map((t) => (
          <div
            key={t.label}
            className="metricTile"
            onMouseEnter={(e) => {
              lastMouseRef.current = { x: e.clientX, y: e.clientY };
              requestShow(t.tipTitle, t.tipBody);
            }}
            onMouseMove={(e) => {
              lastMouseRef.current = { x: e.clientX, y: e.clientY };
              // update position only if tip is already visible (prevents flicker)
              setTip((prev) => (prev ? { ...prev, x: e.clientX, y: e.clientY } : prev));
            }}
            onMouseLeave={hide}
          >
            <div className="metricLabel">{t.label}</div>

            <div className="metricValue">
              {t.value}
              {streaming && t.label === "Run" ? (
                <span style={{ marginLeft: 10, display: "inline-flex", gap: 8, alignItems: "center" }}>
                  <span className="miniSpinner" />
                </span>
              ) : null}
            </div>

            <div className="metricHint">{t.hint}</div>
          </div>
        ))}
      </section>

      {/* Tooltip CSS (scoped via classes, no arrows, glass blur, high z-index) */}
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
          transform: translateY(6px) scale(0.98);
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
      `}</style>
    </>
  );
}

/** ---------- Modal (slider w/ built-in stamps, clickable) ---------- */
function GenerateModal({
  open,
  disabled,
  maxAllowed,
  defaultValue,
  existingCount,
  onClose,
  onConfirm,
}: {
  open: boolean;
  disabled: boolean;
  maxAllowed: number;
  defaultValue: number;
  existingCount: number;
  onClose: () => void;
  onConfirm: (n: number) => void;
}){
  const min = Math.max(1, existingCount);
  const max = clamp(Math.max(maxAllowed, min), min, 500);
  const [value, setValue] = useState(clamp(defaultValue, min, max));

  useEffect(() => {
    if (!open) return;
    setValue(clamp(defaultValue, min, max));
  }, [open, defaultValue, min, max]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "Enter" && !disabled) onConfirm(value);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose, onConfirm, value, disabled]);

  if (!open) return null;

  const stamps = STAMPS.filter((s) => s >= min && s <= max);
  const setToStamp = (s: number) => {
    if (disabled) return;
    setValue(clamp(s, min, max));
  };

  return (
    <div className="modalRoot" role="dialog" aria-modal="true">
      <div className="modalBackdrop" onClick={disabled ? undefined : onClose} />
      <div className="modalCard fadeInUp">
        <div className="modalHeader">
          <div>
            <div className="modalTitle">Generate segments</div>
            <div className="modalSubtitle">
              Choose how many segments to generate. Ticks are shortcuts; you can stop anywhere.
            </div>
          </div>
          <button className="modalClose" onClick={onClose} disabled={disabled} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="modalBody">
          <div style={{ display: "grid", gap: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <div className="muted">Selected</div>
              <div style={{ fontWeight: 900, fontSize: 22 }}>{value}</div>
            </div>

            <input
              type="range"
              className="rsRange"
              min={min}
              max={max}
              value={value}
              disabled={disabled}
              style={{
                ["--pct" as any]: `${((value - min) / Math.max(1, max - min)) * 100}%`,
              }}
              onChange={(e) => setValue(Number(e.target.value))}
            />
            {existingCount > 0 && (
              <div className="helpText">
                You already have <b>{existingCount}</b> segments. Move the slider above {existingCount} to generate more.
              </div>
            )}

            <div className="rsLabels" aria-hidden={false}>
              {stamps.map((s) => {
                const p = ((s - min) / Math.max(1, max - min)) * 100;
                return (
                  <button
                    key={s}
                    type="button"
                    className="rsLabelBtn"
                    style={{ left: `${p}%` }}
                    onClick={() => setToStamp(s)}
                    disabled={disabled}
                    aria-label={`Set segments to ${s}`}
                    title={`Set to ${s}`}
                  >
                    {s}
                  </button>
                );
              })}
            </div>

            <style>{`
              .rsLabels{
                position: relative;
                height: 22px;
                margin-top: 2px;
              }
              .rsLabelBtn{
                position: absolute;
                top: 0;
                transform: translateX(-50%);
                background: transparent;
                border: none;
                padding: 0;
                color: rgba(255,255,255,0.70);
                font-weight: 800;
                font-size: 12px;
                letter-spacing: 0.2px;
                cursor: pointer;
                transition: color 140ms ease, transform 140ms ease;
              }
              .rsLabelBtn:hover{
                color: rgba(255,255,255,0.92);
                transform: translateX(-50%) translateY(-1px);
              }
              .rsLabelBtn:disabled{
                opacity: 0.45;
                cursor: default;
              }
            `}</style>

            <div className="modalPlayRow">
              <button className="playBtnGreen" disabled={disabled} onClick={() => onConfirm(value)} type="button">
                <span className="playIcon">▶</span>
                <span>{disabled ? "Generating…" : "Start"}</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ConfirmResetModal({
  open,
  disabled,
  onClose,
  onConfirm,
}: {
  open: boolean;
  disabled: boolean;
  onClose: () => void;
  onConfirm: () => void;
}) {
  if (!open) return null;

  return (
    <div className="modalRoot" role="dialog" aria-modal="true">
      <div className="modalBackdrop" onClick={disabled ? undefined : onClose} />

      <div className="modalCard fadeInUp" style={{ maxWidth: 520 }}>
        <div className="modalHeader">
          <div>
            <div className="modalTitle">Reset last run?</div>
            <div className="modalSubtitle">
              This clears the persisted <b>last run</b> for you. You can still generate fresh segments anytime.
            </div>
          </div>

          <button className="modalClose" onClick={onClose} disabled={disabled} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="modalBody">
          <div className="resetModalActions">
            <button className="btnGhost" type="button" onClick={onClose} disabled={disabled}>
              Cancel
            </button>

            <button className="btnDangerSoft" type="button" onClick={onConfirm} disabled={disabled}>
              {disabled ? "Resetting…" : "Reset"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
function SettingsModal({
  open,
  disabled,
  canEdit,
  settings,
  onChange,
  onClose,
}: {
  open: boolean;
  disabled: boolean; // e.g. streaming
  canEdit: boolean;  // fresh run only
  settings: UiSettings;
  onChange: (next: Partial<UiSettings>) => void;
  onClose: () => void;
}) {
  if (!open) return null;

  const ToggleRow = ({
    label,
    desc,
    value,
    onToggle,
  }: {
    label: string;
    desc: string;
    value: boolean;
    onToggle: () => void;
  }) => {
    const locked = disabled || !canEdit;

    return (
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: 14,
          alignItems: "flex-start",
          padding: "12px 12px",
          borderRadius: 14,
          border: "1px solid rgba(255,255,255,0.10)",
          background: "rgba(0,0,0,0.14)",
          opacity: locked ? 0.6 : 1,
        }}
      >
        <div style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 900, fontSize: 13, color: "rgba(255,255,255,0.92)" }}>
            {label}
          </div>
          <div style={{ marginTop: 4, fontSize: 12.5, lineHeight: 1.35, color: "rgba(255,255,255,0.65)" }}>
            {desc}
          </div>
        </div>

        <button
          type="button"
          disabled={locked}
          onClick={onToggle}
          className={`toggleBtn ${value ? "toggleOn" : "toggleOff"}`}
          aria-pressed={value}
          title={locked ? "Only editable before a run starts" : value ? "On" : "Off"}
        >
          <span className="toggleKnob" />
        </button>
      </div>
    );
  };

  const lockedBanner = !canEdit ? (
    <div
      style={{
        border: "1px solid rgba(255,255,255,0.10)",
        background: "rgba(255,255,255,0.04)",
        borderRadius: 14,
        padding: "10px 12px",
        color: "rgba(255,255,255,0.70)",
        fontSize: 12.5,
        lineHeight: 1.35,
      }}
    >
      Settings are locked because a run is already loaded. Reset / clear runs to edit settings for a fresh run.
    </div>
  ) : null;

  return (
    <div className="modalRoot" role="dialog" aria-modal="true">
      <div className="modalBackdrop" onClick={disabled ? undefined : onClose} />
      <div className="modalCard fadeInUp" style={{ maxWidth: 720 }}>
        <div className="modalHeader">
          <div>
            <div className="modalTitle">Settings</div>
            <div className="modalSubtitle">Applies to the next fresh run (saved in this browser).</div>
          </div>
          <button className="modalClose" onClick={onClose} disabled={disabled} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="modalBody" style={{ display: "grid", gap: 12 }}>
          {lockedBanner}

          <ToggleRow
            label="LLM descriptions"
            desc="Generate segment descriptions during the run."
            value={settings.enableDescriptions}
            onToggle={() => onChange({ enableDescriptions: !settings.enableDescriptions })}
          />
          <ToggleRow
            label="Pricing model"
            desc="Fill CPM/CPC pricing using the trained model."
            value={settings.enablePricing}
            onToggle={() => onChange({ enablePricing: !settings.enablePricing })}
          />
          <ToggleRow
            label="Taxonomy enrichment"
            desc="Run taxonomy model pass to enrich L1/L2."
            value={settings.enableTaxonomy}
            onToggle={() => onChange({ enableTaxonomy: !settings.enableTaxonomy })}
          />
          <ToggleRow
            label="Coverage shortcut"
            desc="Skip proposals when a close Cybba match already exists."
            value={settings.enableCoverage}
            onToggle={() => onChange({ enableCoverage: !settings.enableCoverage })}
          />
          <ToggleRow
            label="LLM Assistance"
            desc="Analyze competitor catalog gaps and generate unique new segments using Ollama. Runs alongside the regular pipeline."
            value={settings.enableLlmWebAssistance}
            onToggle={() => onChange({ enableLlmWebAssistance: !settings.enableLlmWebAssistance })}
          />

          <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, marginTop: 4 }}>
            <button className="btnGhost" type="button" onClick={onClose} disabled={disabled}>
              Done
            </button>
          </div>
        </div>
      </div>

      <style>{`
        .toggleBtn{
          width: 48px;
          height: 28px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.18);
          background: rgba(255,255,255,0.08);
          cursor: pointer;
          padding: 0;
          position: relative;
          flex: 0 0 auto;
          transition: background 160ms ease, border-color 160ms ease, opacity 160ms ease;
        }
        .toggleBtn:disabled{
          opacity: 0.55;
          cursor: not-allowed;
        }
        .toggleKnob{
          position: absolute;
          top: 3px;
          left: 3px;
          width: 22px;
          height: 22px;
          border-radius: 999px;
          background: rgba(255,255,255,0.92);
          transition: transform 180ms ease;
        }
        .toggleOn{
          background: rgba(34,197,94,0.22);
          border-color: rgba(34,197,94,0.35);
        }
        .toggleOn .toggleKnob{
          transform: translateX(20px);
        }
        .toggleOff{
          background: rgba(255,255,255,0.08);
          border-color: rgba(255,255,255,0.18);
        }
      `}</style>
    </div>
  );
}


/** ---------- App ---------- */
export default function App() {
  const [streaming, setStreaming] = useState(false);

  const [availableValidated, setAvailableValidated] = useState<number | null>(null);
  const [targetRows, setTargetRows] = useState<number>(100);

  const [runId, setRunId] = useState<string | null>(null);
  const [summary, setSummary] = useState<RunSummary | null>(null);

  const [showResetInfo, setShowResetInfo] = useState(false);
  const [progress, setProgress] = useState(0);

  const [revealRows, setRevealRows] = useState(false);
  const revealRef = useRef(false);
  
  // --- prefill progress timer ---
  const prefillTimerRef = useRef<number | null>(null);

  // ✅ refs so progress updates don’t lag
  const progressRef = useRef(0);
  const rowCountForProgressRef = useRef(0);

  const PAGE_SIZE = 25;
  const [page, setPage] = useState(1);
  const [pageAnimKey, setPageAnimKey] = useState(0);

  const [lastRunAt, setLastRunAt] = useState<number | null>(null);
  const [lastRunMeta, setLastRunMeta] = useState<{ run_id: string | null; updated_at: number | null } | null>(null);

  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [resetting, setResetting] = useState(false);

  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const [finalRows, setFinalRows] = useState<Record<string, any>[]>([]);

  const [error, setError] = useState<string | null>(null);
  const noRunAndNoSegments = !runId && rows.length === 0 && finalRows.length === 0;
  const disableComparisonAnalytics = streaming || noRunAndNoSegments;

  const [showGenerateModal, setShowGenerateModal] = useState(false);

  const [query, setQuery] = useState("");
  const [minUniq, setMinUniq] = useState<number>(0);

  const [phase, setPhase] = useState<string>("");

  const stopRef = useRef<null | (() => void)>(null);
  const tokenRef = useRef(0);
  const targetRef = useRef(targetRows);
  const rowsRef = useRef(rows);
  const finalRef = useRef(finalRows);
  const seenRef = useRef<Set<string>>(new Set());

  const pricePollTokenRef = useRef(0);

  const revealTimerRef = useRef<number | null>(null);
  // show overlay while streaming AND we still have nothing to show yet

  const [activePage, setActivePage] = useState<
    "segments" | "cybba" | "comparison" | "analytics" | "suggestions"
  >("segments");  const [pageKey, setPageKey] = useState(0);
  // Add near other state declarations



const SETTINGS_KEY = "cybba_ui_settings_v1";



function loadUiSettings(): UiSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) throw new Error("no settings");
    const obj = JSON.parse(raw);
    return {
      enableDescriptions: obj.enableDescriptions ?? true,
      enablePricing: obj.enablePricing ?? true,
      enableTaxonomy: obj.enableTaxonomy ?? true,
      enableCoverage: obj.enableCoverage ?? true,
      enableLlmWebAssistance: obj.enableLlmWebAssistance ?? false,
    };
  } catch {
    return {
      enableDescriptions: true,
      enablePricing: true,
      enableTaxonomy: true,
      enableCoverage: true,
      enableLlmWebAssistance: false,
    };
  }
}

function saveUiSettings(s: UiSettings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
}

const [showSettingsModal, setShowSettingsModal] = useState(false);
const [uiSettings, setUiSettings] = useState<UiSettings>(() => loadUiSettings());

useEffect(() => {
  saveUiSettings(uiSettings);
}, [uiSettings]);


const isFreshRun = !runId && rows.length === 0 && finalRows.length === 0 && !streaming;

  useEffect(() => {
    setPageKey((k) => k + 1);
  }, [activePage]);

  useEffect(() => { revealRef.current = revealRows; }, [revealRows]);

  useEffect(() => {
    targetRef.current = targetRows;
  }, [targetRows]);

  useEffect(() => {
    rowsRef.current = rows;
  }, [rows]);

  useEffect(() => {
    finalRef.current = finalRows;
  }, [finalRows]);

  // ✅ Rehydrate after laptop sleep / tab hidden -> visible
  useEffect(() => {
    let cancelled = false;

    const rehydrate = async () => {
      if (document.visibilityState !== "visible") return;
      if (!runId) return;

      try {
        const res = await getRun(runId);
        if (cancelled) return;

        setSummary(res.summary ?? null);
        setRows(res.rows ?? []);
        setRevealRows(true);

        if (Array.isArray((res as any)?.final_rows) && (res as any).final_rows.length > 0) {
          setFinalRows((res as any).final_rows.slice(0, targetRef.current));
          mergePricesIntoRows((res as any).final_rows);
        } else {
          // pricing may arrive later
          fetchPricesForRun(runId);
        }
      } catch {
        // ignore (run may not exist yet / backend restarting)
      }
    };

    document.addEventListener("visibilitychange", rehydrate);
    window.addEventListener("focus", rehydrate);

    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", rehydrate);
      window.removeEventListener("focus", rehydrate);
    };
  }, [runId]);



  async function ensureNotifyPermission(): Promise<boolean> {
  if (!("Notification" in window)) return false;
  if (Notification.permission === "granted") return true;
  if (Notification.permission === "denied") return false;
  const perm = await Notification.requestPermission();
  return perm === "granted";
}

function notifyDone(runId: string, rows: number) {
  if (!("Notification" in window)) return;
  if (Notification.permission !== "granted") return;

  const n = new Notification("Segment Expansion — Done", {
    body: `Generated ${rows} rows (run ${runId}).`,
    // icon: "/favicon.ico", // optional
  });

  n.onclick = () => {
    window.focus();
    n.close();
  };
}

  // ✅ helper so progress always clamps and updates ref
  function setProgressSafe(next: number) {
    const v = clamp(Math.round(next), 0, 100);
    progressRef.current = v;
    setProgress(v);
  }

  function stopPrefill() {
  if (prefillTimerRef.current) {
    window.clearInterval(prefillTimerRef.current);
    prefillTimerRef.current = null;
  }
}

  function stopRevealTimer() {
    if (revealTimerRef.current) {
      window.clearTimeout(revealTimerRef.current);
      revealTimerRef.current = null;
    }
  }
  function startPrefillTo99() {
  stopPrefill();
  stopRevealTimer();

  setRevealRows(false);
  setProgressSafe(0);

  const start = Date.now();
  const DURATION_MS = 6000;
  const MAX = 99;

  prefillTimerRef.current = window.setInterval(() => {
    const elapsed = Date.now() - start;
    const t = Math.min(1, elapsed / DURATION_MS);
    const eased = 1 - Math.pow(1 - t, 3);
    let next = Math.floor(eased * MAX);

    // ✅ once we "arrive", keep it subtly alive (96–99)
    if (t >= 1) {
      next = 96 + Math.floor((elapsed / 350) % 4); // 96,97,98,99 loop
    }

    setProgressSafe(next);
  }, 50);
}
  useEffect(() => {
    return () => {
      stopPrefill();
      stopRevealTimer();
    };
  }, []);

  // ✅ NEW: hydrate from persisted "last run" on page load (per user)
  const LAST_RUN_TTL_DAYS = 5;
  const LAST_RUN_TTL_SECONDS = LAST_RUN_TTL_DAYS * 24 * 60 * 60;

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const last = await getLastRun();
        if (cancelled) return;

        setLastRunMeta(last);
        if (typeof last?.updated_at === "number") setLastRunAt(last.updated_at);

        localStorage.setItem("cybba_last_run_shadow", JSON.stringify(last));

        const lastRunId = last?.run_id ?? null;
        if (!lastRunId) return;

        const updatedAt = typeof last.updated_at === "number" ? last.updated_at : null;
        const ageOk =
          updatedAt == null ? true : Math.floor(Date.now() / 1000) - updatedAt <= LAST_RUN_TTL_SECONDS;

        if (!ageOk) return;

        setRunId(lastRunId);

        const res = await getRun(lastRunId);
        if (cancelled) return;

        setSummary(res.summary ?? null);

        // ✅ load all rows
        const loadedRows = Array.isArray(res.rows) ? res.rows : [];
        setRows(loadedRows);

        // ✅ IMPORTANT: sync the UI “display cap” to what we actually loaded
        const loadedCount = loadedRows.length;
        if (loadedCount > 0) {
          const n = clamp(loadedCount, 1, 500); // keep within UI’s supported max
          setTargetRows(n);
          targetRef.current = n; // needed immediately for slicing below
        }

        // (Optional but useful): update slider max from summary if present
        const vt = Number((res as any)?.summary?.validated_total);
        if (Number.isFinite(vt) && vt > 0) setAvailableValidated(vt);

        setRevealRows(true);

        const final = (res as any)?.final_rows;
        if (Array.isArray(final) && final.length > 0) {
          setFinalRows(final.slice(0, targetRef.current));
          mergePricesIntoRows(final);
        } else {
          fetchPricesForRun(lastRunId);
        }
      } catch {
        try {
          const s = localStorage.getItem("cybba_last_run_shadow");
          if (s) {
            const shadow = JSON.parse(s);
            setLastRunMeta(shadow);
            if (typeof shadow?.updated_at === "number") setLastRunAt(shadow.updated_at);
          }
        } catch {}
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function stopStream() {
    if (stopRef.current) stopRef.current();
    stopRef.current = null;
  }

  const baseRows = rows;
  const gatedRows = useMemo(
    () => (revealRows ? rows : []),
    [rows, revealRows]
  );
  const displayRows = useMemo(
    () => gatedRows.slice(0, targetRows),
    [gatedRows, targetRows]
  );
  const hasRows = displayRows.length > 0;

  const filteredRows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return displayRows.filter((r) => {
      const name = String(r["New Segment Name"] ?? r["Proposed New Segment Name"] ?? "").toLowerCase();
      const okQ = !q || name.includes(q);

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

  // -------- Pagination (25 per page) --------
  useEffect(() => {
    setPage(1);
  }, [query, minUniq, targetRows, runId]);

  const totalPages = useMemo(() => Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE)), [filteredRows.length]);
  const safePage = Math.min(page, totalPages);

  useEffect(() => {
    if (safePage !== page) setPage(safePage);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [totalPages]);

  const pagedRows = useMemo(() => {
    const start = (safePage - 1) * PAGE_SIZE;
    return filteredRows.slice(start, start + PAGE_SIZE);
  }, [filteredRows, safePage]);


  function mergePricesIntoRows(priced: Record<string, any>[]) {
    const map = new Map<string, Record<string, any>>();
    for (const r of priced) {
      const k = rowKey(r);
      if (k) map.set(k, r);
    }

    setRows((prev) =>
      prev.map((r) => {
        const k = rowKey(r);
        if (!k) return r;
        const pricedRow = map.get(k);
        if (!pricedRow) return r;
        return { ...r, ...pricedRow };
      })
    );
  }

  async function fetchPricesForRun(runIdToFetch: string) {
    pricePollTokenRef.current += 1;
    const myPollToken = pricePollTokenRef.current;

    const attempts = 40;
    const delayMs = 500;

    for (let i = 0; i < attempts; i++) {
      if (myPollToken !== pricePollTokenRef.current) return;

      try {
        const res = await getRun(runIdToFetch);

        if (Array.isArray((res as any)?.final_rows) && (res as any).final_rows.length > 0) {
          setFinalRows((res as any).final_rows.slice(0, targetRef.current));
          mergePricesIntoRows((res as any).final_rows);
          return;
        }
      } catch {}

      await new Promise((r) => setTimeout(r, delayMs));
    }
  }

  function startOrExtendTo(nextTarget: number) {
    const seed = rowsRef.current;
    seenRef.current = new Set(seed.map(rowKey).filter(Boolean));

    if (seed.length >= nextTarget) {
      stopStream();
      setStreaming(false);
      return;
    }

    stopStream();
    setStreaming(true);
    stopRevealTimer();
    stopPrefill();
    setProgressSafe(0);

    // ✅ init progress
    rowCountForProgressRef.current = seed.length;
    

    setError(null);

    tokenRef.current += 1;
    const myToken = tokenRef.current;

    const stop = runPipelineStream({
      max_rows: nextTarget,
      settings: uiSettings,

      onRunId: (id) => {
        if (myToken !== tokenRef.current) return;
        setRunId(id);
      },

      onSummary: (s) => {
        if (myToken !== tokenRef.current) return;

        setSummary(s);

        const vt = Number((s as any)?.validated_total);
        if (Number.isFinite(vt) && vt > 0) setAvailableValidated(vt);

        const phase = String((s as any)?.phase ?? "");
        setPhase(phase); // <-- add this if you created phase state

        // ✅ when backend switches to streaming rows, finish + reveal
        if (phase === "streaming_rows") {
          stopPrefill();
          setProgressSafe(100);

          stopRevealTimer();
          revealTimerRef.current = window.setTimeout(() => {
            setRevealRows(true);
            revealTimerRef.current = null;
          }, 150);

          return;
        }

        // ✅ otherwise keep using real progress when available
        const cur = Number((s as any)?.current);
        const total = Number((s as any)?.total);

        if (Number.isFinite(cur) && Number.isFinite(total) && total > 0) {
          const pct = (cur / total) * 100;
          setProgressSafe(phase === "done" ? 100 : Math.min(99, pct));
        }
      },

      onRow: (row) => {
        if (myToken !== tokenRef.current) return;

        const limit = targetRef.current;
        const k = rowKey(row);
        if (!k || seenRef.current.has(k)) return;
        seenRef.current.add(k);

        rowCountForProgressRef.current = seenRef.current.size;

        // ✅ FIRST accepted row: finish the bar, then reveal rows
        if (!revealRef.current) {
          stopPrefill();
          stopRevealTimer();

          //setProgressSafe(100);

          revealTimerRef.current = window.setTimeout(() => {
            setRevealRows(true);
            revealTimerRef.current = null;
          }, 200);
        }

        setRows((prev) => {
          if (prev.length >= limit) return prev;
          return [...prev, row];
        });
      },

      onDone: (payload: any) => {
        if (myToken !== tokenRef.current) return;

        stopPrefill();
        setProgressSafe(100);

        if (!revealRef.current) {
          stopRevealTimer();
          revealTimerRef.current = window.setTimeout(() => {
            setRevealRows(true);
            revealTimerRef.current = null;
          }, 250);
        }
        if (payload?.summary) setSummary(payload.summary);
        if (payload?.run_id) setRunId(payload.run_id);

        if (Array.isArray(payload?.final_rows) && payload.final_rows.length > 0) {
          setFinalRows(payload.final_rows.slice(0, targetRef.current));
          mergePricesIntoRows(payload.final_rows);
        } else if (payload?.run_id) {
          fetchPricesForRun(payload.run_id);
        }
        if (payload?.run_id) {
          notifyDone(payload.run_id, targetRef.current);
        }
        // ✅ let 100% sit briefly, then hide overlay + reset
        setTimeout(() => {
          setStreaming(false);
          setProgressSafe(0);
        }, 450);

        stopRef.current = null;
      },

      onError: (msg) => {
        if (myToken !== tokenRef.current) return;
        stopPrefill();
        setError(msg);
        setStreaming(false);
        setProgressSafe(0);
        stopRef.current = null;
      },
    });

    stopRef.current = stop;
  }

  function onGenerateSelected(n: number) {
    const maxAllowed = availableValidated ?? 500;
    const nextTarget = clamp(n, 1, maxAllowed);

    if (nextTarget < targetRef.current) {
      stopStream();
      setStreaming(false);
    }

    setTargetRows(nextTarget);
    targetRef.current = nextTarget;

    startOrExtendTo(nextTarget);
  }

  function onDownload(mode: Mode) {
    if (!runId) return;
    downloadCsv(runId, mode);
  }

  const showOverlay = streaming && phase !== "streaming_rows";  //const showOverlay = streaming && rows.length === 0;
  const modalMax = availableValidated ?? 500;

  const progressLabel = useMemo(() => {
    const p = progress;
    if (p < 15) return "Warming up the engines…";
    if (p < 35) return "Loading the good stuff…";
    if (p < 70) return "Generating your latest segments — hold tight…";
    if (p < 92) return "Adding finishing touches…";
    if (p < 100) return "Final polish…";
    return "Ready ✨";
  }, [progress]);

  const comparisonRows = (finalRows?.length ? finalRows : rows);
  const analyticsRows = rows;

  const EmptyState = (
  <div className="emptyStateStage">
    <div className="emptyStateCard">
      <div className="emptyStateTitle">No segments available</div>
      <div className="emptyStateSub">
        Click{" "}
        <button type="button" className="emptyStateLink" onClick={() => setShowGenerateModal(true)}>
          Generate
        </button>{" "}
        to build new segments.
      </div>
    </div>
  </div>
  );
  const noSegmentsAvailable = !streaming && (comparisonRows?.length ?? 0) === 0;
    // ✅ Segments page content EXACTLY as it exists today (no logic changes)
  const SegmentsView = (
    <>
      {hasRows ? <KpiStrip runId={runId} rows={displayRows} summary={summary} streaming={streaming} /> : null}

      <div className={`toast ${error ? "toastShow" : ""}`}>
        <div className="toastTitle">Backend error</div>
        <div className="toastBody">{error ?? ""}</div>
      </div>

      {hasRows ? (
        <section className="filtersStrip fadeInUp">
          <div className="filtersStripHeader">
            <div className="cardTitle">Filters</div>
            <div className="cardHint">
              Showing <b>{filteredRows.length}</b> / {displayRows.length}
            </div>
          </div>

          <div className="filtersStripBody">
            <div className="field">
              <span>Search</span>
              <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search segments..." />
            </div>

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

            <div className="field">
              <span>Status</span>
              <div className="statusPill">
                {streaming ? (
                  <>
                    <span className="miniSpinner" />
                    <span>streaming…</span>
                  </>
                ) : runId ? (
                  <>
                    <span className="pill">run</span>
                    <code style={{ opacity: 0.9 }}>{runId}</code>
                  </>
                ) : (
                  <span className="muted">ready</span>
                )}
              </div>
            </div>
          </div>
        </section>
      ) : null}

      <section className="card fadeInUp">
  <div className="cardHeader">
    <div className="cardTitle">Priced Segments</div>
    {streaming ? (
      <div className="muted" style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
        <span className="miniSpinner" />
        streaming…
      </div>
    ) : null}
  </div>

  <div className="pageFade tableOverlayWrap" key={pageAnimKey}>
    <SegmentsTable rows={pagedRows} loading={streaming && rows.length === 0} />

    {showOverlay && (
      <div className="tableOverlay">
        <div className="overlayCard">
          <div className="overlayText" style={{ flex: 1, minWidth: 0 }}>
            <div className="overlayTitle">Running pipeline</div>
            <div className="overlaySub" style={{ marginTop: 6 }}>
              {progressLabel}
            </div>

            <div
              className="pbWrap"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={progress}
            >
              <div className="pbFill" style={{ width: `${progress}%` }} />
              <div className="pbGlow" style={{ left: `${progress}%` }} />
            </div>

            <div className="pbMetaRow">
              <div className="pbMetaLeft">{Math.round(progress)}%</div>
              <div className="pbMetaRight">
                {Math.min(rowCountForProgressRef.current, targetRef.current)} / {targetRef.current} rows
              </div>
            </div>
          </div>
        </div>
      </div>
    )}
  </div>

  {filteredRows.length > PAGE_SIZE ? (
    <div className="pagerRow">
      <button
        className="pagerBtn"
        disabled={safePage <= 1}
        onClick={() => {
          setPage((p) => Math.max(1, p - 1));
          setPageAnimKey((k) => k + 1);
        }}
      >
        Prev
      </button>

      <div className="pagerNums">
        {Array.from({ length: totalPages }, (_, i) => i + 1)
          .filter((n) => n === 1 || n === totalPages || Math.abs(n - safePage) <= 2)
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
        }}
      >
        Next
      </button>
    </div>
  ) : null}
</section>

      {!streaming && filteredRows.length === 0 ? (
        <div className="emptyStateStage">
          <div className="emptyStateCard">
            <div className="emptyStateTitle">No segments available</div>
            <div className="emptyStateSub">
              Click{" "}
              <button type="button" className="emptyStateLink" onClick={() => setShowGenerateModal(true)}>
                Generate
              </button>{" "}
              to build new segments.
            </div>
          </div>
        </div>
      ) : null}
    </>
  );

  return (
  <ThemeProvider>
  <div className="appShell navPinned">
    <div className="bg" />

  <SideMenu
    active={activePage}
    disabledIds={disableComparisonAnalytics ? ["comparison", "analytics"] : []}
    onSelect={(id) => setActivePage(id as any)}
  />

    <div className="app appMain">

      <GenerateModal
        open={showGenerateModal}
        disabled={streaming}
        maxAllowed={modalMax}
        defaultValue={targetRows}
        existingCount={rows.length}
        onClose={() => (!streaming ? setShowGenerateModal(false) : null)}
        onConfirm={(n) => {
          setShowGenerateModal(false);
          setTimeout(() => onGenerateSelected(n), 60);
        }}
      />

     <SettingsModal
      open={showSettingsModal}
      disabled={streaming}
      canEdit={isFreshRun}
      settings={uiSettings}
      onChange={(patch) => setUiSettings((prev) => ({ ...prev, ...patch }))}
      onClose={() => setShowSettingsModal(false)}
    />

      <ConfirmResetModal
        open={showResetConfirm}
        disabled={streaming || resetting}
        onClose={() => setShowResetConfirm(false)}
        onConfirm={async () => {
          setResetting(true);
          try {
            await resetLastRun();
          } catch {}

          // Clear current UI so next Generate is fresh
          setRunId(null);
          setSummary(null);
          setRows([]);
          setFinalRows([]);
          setAvailableValidated(null);

          setShowResetConfirm(false);
          setResetting(false);
        }}
      />

      

      {showResetInfo ? (
        <div className="modalRoot" role="dialog" aria-modal="true">
          <div className="modalBackdrop" onClick={() => setShowResetInfo(false)} />

          <div className="modalCard fadeInUp" style={{ maxWidth: 560 }}>
            <div className="modalHeader">
              <div>
                <div className="modalTitle">Reset policy</div>
                <div className="modalSubtitle">Last run is saved for quick resume. Here’s how the 5-day rule works.</div>
              </div>

              <button className="modalClose" onClick={() => setShowResetInfo(false)} aria-label="Close">
                ✕
              </button>
            </div>

            <div className="modalBody">
              <div className="resetInfoBody">
                <p>
                  <b>Last run always displays</b> when you open the app, so you can quickly resume or download exports.
                </p>

                <p>
                  If your last run is <b>within 5 days</b>, the UI may keep showing that run unless you generate more
                  than what’s already loaded.
                </p>

                <p>
                  If the last run is <b>older than 5 days</b>, the app does <b>not</b> hydrate the old run — the next
                  Generate will start a <b>fresh run from scratch</b>.
                </p>

                <p style={{ marginBottom: 0 }}>
                  Use <b>Reset</b> anytime to clear the persisted pointer and generate fresh immediately.
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      <header className="header">
        <div className="titleBlock">   
          <img className="brandLogo" src={cybbaLogo} alt="Cybba logo" />
          <h1 className="title">Segment Expansion</h1>
          <p className="subtitle">Generate, price, and export new audience segments.</p>

          {lastRunAt ? (
            <div className="lastRunRow">
              <span className="lastRunLabel">LAST RUN</span>
              <span className="lastRunValue">{new Date(lastRunAt * 1000).toLocaleString()}</span>

              <button
                className="resetInlineBtn"
                onClick={() => setShowResetConfirm(true)}
                disabled={streaming}
                title="Reset last run"
              >
                ⟲ Reset
              </button>

              <button
                type="button"
                className="resetInfoBtn"
                onClick={() => setShowResetInfo(true)}
                aria-label="Reset policy info"
                title="How reset works"
              >
                i
              </button>
            </div>
          ) : null}
        </div>

        <div className="actions">
          <button
            className="iconBtn"
            type="button"
            onClick={() => setShowSettingsModal(true)}
            disabled={streaming}
            title="Settings"
            aria-label="Settings"
          >
            {"\u2699\uFE0E"}
          </button>

          <button
            className="btnPrimary"
            type="button"
            onClick={async () => {
              await ensureNotifyPermission();
              setShowGenerateModal(true);
            }}
            disabled={streaming}
            title="Choose how many segments to generate"
          >
            <span className={`btnSpinner ${streaming ? "btnSpinnerShow" : ""}`} />
            {streaming ? "Generating…" : "Generate"}
          </button>

          <DownloadButton
            disabled={!runId || streaming}
            onDownloadFinal={() => onDownload("final")}
            onDownloadValidated={() => onDownload("validated")}
          />
        </div>
      </header>

      

      <main className="main">
        
        <div key={`${activePage}-${pageKey}`} className="pageStage pageEnter">
          {activePage === "suggestions" ? (
            <SuggestionsPage streaming={streaming} runId={runId} />
          ) :
          activePage === "segments" ? (
            SegmentsView
          ) : activePage === "cybba" ? (
              <CybbaSegmentsPage streaming={streaming}/>
          ) : activePage === "comparison" ? (
            noSegmentsAvailable ? EmptyState : <ComparisonPage rows={comparisonRows} streaming={streaming} />
          ) : activePage === "analytics" ? (
            noSegmentsAvailable ? EmptyState : <AnalyticsPage rows={analyticsRows} streaming={streaming} />
          ) : null}
        </div>

        {/* ✅ smooth transition only for page swap, does not touch existing theme */}
        <style>{`
          .pageStage { will-change: opacity, transform; }
          .pageEnter { animation: pageEnter 220ms ease-out both; }
          @keyframes pageEnter {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0px); }
          }
        `}</style>
      
      </main>

      {/* ✅ NOTE: If you want the progress overlay visible for the whole run (not only rows===0),
          change showOverlay to: const showOverlay = streaming; */}
      {/* ... keep everything you already have in render ... */}

      {/* ✅ Replace your existing overlay block with this */}
      
    </div>
  </div>
  </ThemeProvider>
  );
  
}

