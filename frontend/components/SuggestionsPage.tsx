import React, { useEffect, useRef, useState } from "react";
import {
  generateSuggestions,
  analyzeSuggestion,
  SuggestionItem,
  SuggestionAnalysis,
  SuggestionMatch,
} from "../src/api";

/* ──────────────────────────────────────────────────────────
   Sub-components
────────────────────────────────────────────────────────── */

function MatchPill({ m, type }: { m: SuggestionMatch; type: "competitor" | "cybba" }) {
  const pct = Math.round(m.similarity * 100);
  const styles = {
    competitor: {
      bg: "rgba(59,130,246,0.15)",
      border: "rgba(59,130,246,0.35)",
      color: "rgba(147,197,253,1)",
    },
    cybba: {
      bg: "rgba(16,185,129,0.15)",
      border: "rgba(16,185,129,0.35)",
      color: "rgba(110,231,183,1)",
    },
  }[type];

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        padding: "3px 9px",
        borderRadius: 999,
        background: styles.bg,
        border: `1px solid ${styles.border}`,
        color: styles.color,
        fontSize: 11.5,
        fontWeight: 600,
        whiteSpace: "nowrap",
      }}
    >
      <span style={{ opacity: 0.7 }}>{m.provider}</span>
      <span style={{ opacity: 0.4 }}>·</span>
      <span>{m.segment_name.length > 32 ? m.segment_name.slice(0, 30) + "…" : m.segment_name}</span>
      <span style={{ opacity: 0.6, fontVariantNumeric: "tabular-nums" }}>{pct}%</span>
    </span>
  );
}

function AnalysisBlock({
  analysis,
  loading,
}: {
  analysis: SuggestionAnalysis | null;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "12px 14px",
          background: "rgba(255,255,255,0.04)",
          borderRadius: 10,
          marginTop: 12,
          color: "rgba(255,255,255,0.50)",
          fontSize: 13,
        }}
      >
        <span className="btnSpinner btnSpinnerShow" style={{ width: 13, height: 13, borderWidth: 2 }} />
        Asking Ollama for analysis…
      </div>
    );
  }

  if (!analysis) return null;

  const useful = analysis.is_useful;
  const headerBg =
    useful === true ? "rgba(16,185,129,0.12)" :
    useful === false ? "rgba(239,68,68,0.10)" :
    "rgba(255,255,255,0.04)";
  const icon = useful === true ? "✓" : useful === false ? "✗" : "?";
  const iconColor =
    useful === true ? "rgba(110,231,183,1)" :
    useful === false ? "rgba(252,165,165,1)" :
    "rgba(255,255,255,0.35)";
  const label =
    useful === true ? "Recommended" :
    useful === false ? "Lower priority" :
    "Analysis unavailable";

  return (
    <div
      style={{
        marginTop: 12,
        borderRadius: 10,
        border: "1px solid rgba(255,255,255,0.08)",
        background: "rgba(0,0,0,0.16)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "9px 13px",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
          background: headerBg,
        }}
      >
        <span style={{ fontSize: 14, color: iconColor }}>{icon}</span>
        <span style={{ fontWeight: 700, fontSize: 13, color: iconColor }}>{label}</span>
        {analysis.error && (
          <span style={{ marginLeft: "auto", color: "rgba(252,165,165,0.6)", fontSize: 11 }}>
            {analysis.error}
          </span>
        )}
      </div>

      <div style={{ padding: "11px 13px", display: "grid", gap: 9 }}>
        {analysis.helpfulness_reasoning && (
          <AnalysisRow label="Why it helps" text={analysis.helpfulness_reasoning} />
        )}
        {analysis.scaling_tips && (
          <AnalysisRow label="Scaling tips" text={analysis.scaling_tips} />
        )}
        {analysis.competitor_context && (
          <AnalysisRow label="Competitor context" text={analysis.competitor_context} />
        )}
      </div>
    </div>
  );
}

function AnalysisRow({ label, text }: { label: string; text: string }) {
  return (
    <div style={{ display: "grid", gap: 3 }}>
      <div
        style={{
          fontSize: 10.5,
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.06em",
          color: "rgba(255,255,255,0.30)",
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: 13, color: "rgba(255,255,255,0.78)", lineHeight: 1.5 }}>
        {text}
      </div>
    </div>
  );
}

/* ──────────────────────────────────────────────────────────
   Main component
────────────────────────────────────────────────────────── */

export default function SuggestionsPage({
  streaming,
  runId,
}: {
  streaming: boolean;
  runId?: string | null;
}) {
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [items, setItems] = useState<SuggestionItem[]>([]);
  const [note, setNote] = useState<string | null>(null);

  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [analyses, setAnalyses] = useState<Record<string, SuggestionAnalysis>>({});
  const [analyzingId, setAnalyzingId] = useState<string | null>(null);

  // Track the last runId we loaded for — avoid duplicate calls
  const loadedForRef = useRef<string | null | undefined>("__INIT__");
  const loadingRef = useRef(false); // guard against concurrent calls

  async function load(rid?: string | null, forceRefresh = false) {
    // Prevent concurrent calls
    if (loadingRef.current && !forceRefresh) return;
    // Skip if we already loaded for this exact runId
    if (!forceRefresh && loadedForRef.current === (rid ?? null)) return;

    loadingRef.current = true;
    loadedForRef.current = rid ?? null;

    setLoading(true);
    setErr(null);
    setNote(null);
    setExpandedId(null);
    setAnalyses({});

    try {
      const payload = await generateSuggestions(rid, 25);
      setItems(payload.items || []);
      setNote(payload.note || null);
    } catch (e: any) {
      // Only show error if we have a runId (otherwise empty state is expected)
      if (rid) {
        setErr(e?.message ?? "Failed to load suggestions");
      }
      setItems([]);
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  }

  // Single effect: fires on mount and when runId changes
  useEffect(() => {
    // On mount, always load (even with null runId — shows empty state gracefully)
    // On runId change, only reload if it's a real new run
    const isNewRun = runId != null && loadedForRef.current !== runId;
    const isFirstLoad = loadedForRef.current === "__INIT__";

    if (isFirstLoad || isNewRun) {
      load(runId);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  async function handleCardClick(item: SuggestionItem) {
    const id = item.id;
    if (expandedId === id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(id);
    if (analyses[id]) return; // already cached

    setAnalyzingId(id);
    try {
      const result = await analyzeSuggestion({
        segment_name: item.title,
        competitor_matches: item.competitor_matches ?? [],
        cybba_matches: item.cybba_matches ?? [],
        l1: item.proposed_l1,
        l2: item.proposed_l2,
      });
      setAnalyses((prev) => ({ ...prev, [id]: result }));
    } catch (e: any) {
      setAnalyses((prev) => ({
        ...prev,
        [id]: {
          is_useful: null,
          helpfulness_reasoning: "",
          scaling_tips: "",
          competitor_context: "",
          error: e?.message ?? "Analysis failed",
        },
      }));
    } finally {
      setAnalyzingId(null);
    }
  }

  const webCount = items.filter((it) => it.source === "web_assisted").length;

  return (
    <div className="pageFade">
      {/* ── Header strip ── */}
      <section className="filtersStrip fadeInUp">
        <div className="filtersStripHeader">
          <div className="cardTitle">Suggestions</div>
          <div className="cardHint">
            Coverage-gap analysis — segments with high competitor presence but low Cybba coverage.
            Click a card for AI analysis.
          </div>
        </div>

        <div className="filtersStripBody" style={{ alignItems: "center" }}>
          <div className="field">
            <span>Status</span>
            <div className="statusPill">
              {loading ? (
                <>
                  <span className="miniSpinner" />
                  <span>loading…</span>
                </>
              ) : (
                <span className="muted">
                  {items.length > 0 ? `${items.length} suggestions` : "ready"}
                </span>
              )}
            </div>
          </div>

          {webCount > 0 && (
            <div className="field">
              <span>Web-assisted</span>
              <div className="statusPill">
                <span
                  className="pill"
                  style={{
                    background: "rgba(139,92,246,0.25)",
                    border: "1px solid rgba(139,92,246,0.4)",
                    color: "rgba(167,139,250,1)",
                  }}
                >
                  {webCount}
                </span>
              </div>
            </div>
          )}

          <div style={{ marginLeft: "auto" }}>
            <button
              className="btnGhost"
              type="button"
              disabled={streaming || loading}
              onClick={() => load(runId, true)}
            >
              {loading ? "Loading…" : "Refresh"}
            </button>
          </div>
        </div>

        {err && (
          <div className="toast toastShow" style={{ marginTop: 10 }}>
            <div className="toastTitle">Error</div>
            <div className="toastBody">{err}</div>
          </div>
        )}
      </section>

      {/* ── Note (e.g. no run data) ── */}
      {note && (
        <section
          className="card fadeInUp"
          style={{ marginBottom: 0 }}
        >
          <div style={{ padding: "12px 16px", color: "rgba(255,255,255,0.50)", fontSize: 13 }}>
            {note}
          </div>
        </section>
      )}

      {/* ── Cards ── */}
      <section className="card fadeInUp">
        <div className="cardHeader">
          <div className="cardTitle">Segment Analysis</div>
          <div className="muted">
            Sorted by coverage gap (competitor match − Cybba match). Click any card to get AI analysis.
          </div>
        </div>

        {/* Loading skeleton */}
        {loading && items.length === 0 ? (
          <div style={{ padding: 16, display: "grid", gap: 8 }}>
            {Array.from({ length: 5 }).map((_, i) => (
              <div
                key={i}
                className="skeletonRow"
                style={{ height: 58, borderRadius: 12, opacity: 1 - i * 0.15 }}
              >
                <div className="skeletonBar" style={{ height: "100%", borderRadius: 10 }} />
              </div>
            ))}
          </div>
        ) : items.length === 0 ? (
          <div className="empty" style={{ padding: 24 }}>
            <div className="emptyTitle">No suggestions yet</div>
            <div className="muted" style={{ marginTop: 6 }}>
              {runId
                ? "Click Refresh to reload suggestions for the current run."
                : "Generate segments first, then open Suggestions."}
            </div>
          </div>
        ) : (
          <div style={{ padding: 12, display: "grid", gap: 7 }}>
            {items.map((it) => {
              const isExpanded = expandedId === it.id;
              const isAnalyzing = analyzingId === it.id;
              const analysis = analyses[it.id] ?? null;
              const isWebAssisted = it.source === "web_assisted";

              return (
                <div
                  key={it.id}
                  onClick={() => !streaming && handleCardClick(it)}
                  style={{
                    borderRadius: 14,
                    border: isExpanded
                      ? "1px solid rgba(99,102,241,0.45)"
                      : "1px solid rgba(255,255,255,0.09)",
                    background: isExpanded
                      ? "rgba(99,102,241,0.07)"
                      : "rgba(0,0,0,0.10)",
                    cursor: streaming ? "default" : "pointer",
                    opacity: streaming ? 0.6 : 1,
                    transition: "border-color 0.15s, background 0.15s",
                  }}
                >
                  {/* Card header row */}
                  <div style={{ padding: "13px 15px", display: "grid", gap: 7 }}>
                    <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 7, flexWrap: "wrap" }}>
                        <span style={{ fontWeight: 800, color: "rgba(255,255,255,0.92)", fontSize: 13.5 }}>
                          {it.title}
                        </span>
                        {isWebAssisted && (
                          <span
                            style={{
                              padding: "2px 7px",
                              borderRadius: 999,
                              background: "rgba(139,92,246,0.22)",
                              border: "1px solid rgba(139,92,246,0.4)",
                              color: "rgba(167,139,250,1)",
                              fontSize: 10,
                              fontWeight: 700,
                              letterSpacing: "0.04em",
                            }}
                          >
                            Web-assisted
                          </span>
                        )}
                      </div>
                      <span
                        style={{
                          color: "rgba(255,255,255,0.30)",
                          fontSize: 11,
                          flexShrink: 0,
                          marginTop: 2,
                          transform: isExpanded ? "rotate(180deg)" : "none",
                          transition: "transform 0.2s",
                        }}
                      >
                        ▾
                      </span>
                    </div>

                    {it.why && (
                      <div style={{ color: "rgba(255,255,255,0.55)", fontSize: 12.5, lineHeight: 1.4 }}>
                        {it.why}
                      </div>
                    )}

                    {(it.proposed_l1 || it.proposed_l2) && (
                      <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
                        {it.proposed_l1 && <span className="pill">{it.proposed_l1}</span>}
                        {it.proposed_l2 && <span className="pill">{it.proposed_l2}</span>}
                      </div>
                    )}

                    {/* Collapsed preview of top matches */}
                    {!isExpanded && (
                      <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                        {(it.competitor_matches ?? []).slice(0, 2).map((m) => (
                          <MatchPill key={m.segment_name + m.provider} m={m} type="competitor" />
                        ))}
                        {(it.cybba_matches ?? []).slice(0, 1).map((m) => (
                          <MatchPill key={m.segment_name} m={m} type="cybba" />
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Expanded body */}
                  {isExpanded && (
                    <div
                      style={{
                        padding: "0 15px 14px",
                        borderTop: "1px solid rgba(255,255,255,0.06)",
                        paddingTop: 13,
                      }}
                    >
                      <div style={{ display: "grid", gap: 10 }}>
                        {(it.competitor_matches?.length ?? 0) > 0 && (
                          <div>
                            <div
                              style={{
                                fontSize: 10.5,
                                fontWeight: 700,
                                textTransform: "uppercase",
                                letterSpacing: "0.06em",
                                color: "rgba(147,197,253,0.65)",
                                marginBottom: 5,
                              }}
                            >
                              Competitor matches
                            </div>
                            <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                              {(it.competitor_matches ?? []).slice(0, 6).map((m) => (
                                <MatchPill key={m.segment_name + m.provider} m={m} type="competitor" />
                              ))}
                            </div>
                          </div>
                        )}

                        {(it.cybba_matches?.length ?? 0) > 0 ? (
                          <div>
                            <div
                              style={{
                                fontSize: 10.5,
                                fontWeight: 700,
                                textTransform: "uppercase",
                                letterSpacing: "0.06em",
                                color: "rgba(110,231,183,0.65)",
                                marginBottom: 5,
                              }}
                            >
                              Cybba catalog (existing)
                            </div>
                            <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                              {(it.cybba_matches ?? []).slice(0, 6).map((m) => (
                                <MatchPill key={m.segment_name} m={m} type="cybba" />
                              ))}
                            </div>
                          </div>
                        ) : (
                          <div style={{ fontSize: 12, color: "rgba(110,231,183,0.55)", fontStyle: "italic" }}>
                            Not found in Cybba catalog — this is a coverage gap.
                          </div>
                        )}
                      </div>

                      <AnalysisBlock analysis={analysis} loading={isAnalyzing} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
