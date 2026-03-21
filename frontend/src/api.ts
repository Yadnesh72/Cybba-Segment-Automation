/**
 * Central API helper for Cybba Segment Expansion
 * Talks to FastAPI backend
 */

export const API_BASE =
  (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8000";

/* =========================
   Types
========================= */

export type QuantileStats = {
  min: number;
  p10: number;
  p50: number;
  p90: number;
  mean: number;
};

export interface RunSummary {
  total_proposals: number;
  validated?: number; // legacy/non-capped runs
  covered: number;

  // capping support
  validated_total?: number; // before cap
  validated_generated?: number; // after cap
  cap_applied?: number | null;

  duplicates_within_output_normalized?: number;

  dropped_naming: number;
  dropped_underived_only?: number;
  dropped_net_new?: number;
  collisions_resolved: number;

  uniqueness_stats?: QuantileStats;
  rank_stats?: QuantileStats;

  phase?: "streaming_rows" | "done";
  [key: string]: any;
}

export interface ValidatedRow {
  "Proposed New Segment Name": string;
  "Non Derived Segments utilized": string;
  "Composition Similarity"?: number;
  "Closest Cybba Similarity"?: number;
  "Segment Description"?: string;
  uniqueness_score?: number;
  rank_score?: number;
  [key: string]: any;
}

/**
 * Final output row (template-driven, includes pricing columns).
 * Keep it loose because template may add/remove columns.
 */
export type FinalRow = Record<string, any>;

/** When streaming, the backend may emit validated rows (row) and/or final rows (final_row) */
export type StreamRowPayload = Record<string, any>;

export interface RunResponse {
  run_id: string;
  summary: RunSummary;
  rows: ValidatedRow[];
  // optional if you extend non-stream endpoint to return final rows too
  final_rows?: FinalRow[];
}

/**
 * Streaming done payload can include final_rows so the UI can show pricing.
 */
export type StreamDonePayload = {
  run_id: string;
  summary: RunSummary;
  rows?: ValidatedRow[];
  final_rows?: FinalRow[];
};

export type StreamErrorPayload = {
  message: string;
  run_id?: string;
};

/** ✅ NEW: last-run response */
export type LastRunResponse = {
  run_id: string | null;
  updated_at: number | null; // epoch seconds (or null)
  user_id?: string;
};

export type UiSettings = {
  enableDescriptions: boolean;
  enablePricing: boolean;
  enableTaxonomy: boolean;
  enableCoverage: boolean;
  enableLlmWebAssistance: boolean;
};

export type SuggestionMatch = {
  provider: string;
  segment_name: string;
  similarity: number;
  description?: string | null;
};

export type SuggestionItem = {
  id: string;
  title: string;
  why?: string;
  proposed_l1?: string;
  proposed_l2?: string;
  seed_keywords?: string[];
  seed_leaves?: string[];
  competitor_matches?: SuggestionMatch[];
  cybba_matches?: SuggestionMatch[];
  source?: "regular" | "web_assisted";
};

export type SuggestionPayload = {
  suggestion_set_id: string;
  created_at: number;
  items: SuggestionItem[];
  note?: string;
};

export type SuggestionAnalysis = {
  is_useful: boolean | null;
  helpfulness_reasoning: string;
  scaling_tips: string;
  competitor_context: string;
  error?: string;
};

/* =========================
   Helpers
========================= */

async function readError(res: Response) {
  const text = await res.text().catch(() => "");
  return text || `Request failed (${res.status})`;
}

/* =========================
   Per-user: last run
========================= */

/** ✅ NEW: get last run id for current user */
export async function getLastRun(): Promise<LastRunResponse> {
  const res = await fetch(`${API_BASE}/api/users/me/last-run`, {
    method: "GET",
    credentials: "include",
  });

  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function resetLastRun(): Promise<{ ok: boolean }> {
  const res = await fetch(`${API_BASE}/api/users/me/last-run`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
export async function getCybbaCatalog(params: { page: number; page_size: number; q?: string }) {
  const qs = new URLSearchParams();
  qs.set("page", String(params.page));
  qs.set("page_size", String(params.page_size));
  if (params.q) qs.set("q", params.q);

  const res = await fetch(`${API_BASE}/api/catalog/cybba?${qs.toString()}`);
  if (!res.ok) throw new Error(`Catalog request failed (${res.status})`);
  return res.json(); // expected: { rows, total, page }
}
/** ✅ OPTIONAL: explicitly set last run id for current user */
export async function setLastRun(runId: string): Promise<LastRunResponse> {
  const res = await fetch(`${API_BASE}/api/users/me/last-run`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId }),
  });

  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

/* =========================
   API calls (non-stream)
========================= */

export async function runPipeline(max_rows?: number): Promise<RunResponse> {
  const url =
    typeof max_rows === "number" && max_rows > 0
      ? `${API_BASE}/api/run?max_rows=${encodeURIComponent(max_rows)}`
      : `${API_BASE}/api/run`;

  const res = await fetch(url, { method: "POST" });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Failed to run pipeline");
  }

  return res.json();
}

export async function getRun(runId: string): Promise<RunResponse> {
  const res = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(runId)}`);

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Failed to fetch run");
  }

  return res.json();
}

/* =========================
   API calls (streaming / SSE)
========================= */

export function runPipelineStream(opts: {
  max_rows?: number; // backend expects this name
  onRunId?: (runId: string) => void;
  onSummary?: (summary: RunSummary) => void;
  settings?: UiSettings;

  /**
   * row event = validated row (today)
   * Keep loose so UI doesn't break if backend adds pricing cols later.
   */
  onRow?: (row: StreamRowPayload) => void;

  /**
   * optional future event if you later stream final/priced rows separately
   * (not required for current backend)
   */
  onFinalRow?: (row: FinalRow) => void;

  onDone?: (payload: StreamDonePayload) => void;
  onError?: (message: string) => void;
  onOpen?: () => void;
}): () => void {
  const maxRows =
    typeof opts.max_rows === "number" && opts.max_rows > 0
      ? opts.max_rows
      : undefined;

const params = new URLSearchParams();

if (maxRows !== undefined) {
  params.set("max_rows", String(maxRows));
}

// ✅ settings -> query params
if (opts.settings) {
  params.set("enable_descriptions", String(opts.settings.enableDescriptions));
  params.set("enable_pricing", String(opts.settings.enablePricing));
  params.set("enable_taxonomy", String(opts.settings.enableTaxonomy));
  params.set("enable_coverage", String(opts.settings.enableCoverage));
  params.set("enable_llm_web_assistance", String(opts.settings.enableLlmWebAssistance));
}

const url =
  params.toString().length > 0
    ? `${API_BASE}/api/run/stream?${params.toString()}`
    : `${API_BASE}/api/run/stream`;

  // EventSource is GET-only (matches backend)
  const es = new EventSource(url);

  const safeParse = (s: string) => {
    try {
      return JSON.parse(s);
    } catch {
      return null;
    }
  };

  es.onopen = () => opts.onOpen?.();

  es.addEventListener("run_id", (e: MessageEvent) => {
    const payload = safeParse(e.data);
    if (payload?.run_id) opts.onRunId?.(payload.run_id);
  });

  es.addEventListener("summary", (e: MessageEvent) => {
    const payload = safeParse(e.data);
    if (payload) opts.onSummary?.(payload as RunSummary);
  });

  // validated rows (current backend behavior)
  es.addEventListener("row", (e: MessageEvent) => {
    const payload = safeParse(e.data);
    if (payload) opts.onRow?.(payload as StreamRowPayload);
  });

  // optional future event if you emit priced rows
  es.addEventListener("final_row", (e: MessageEvent) => {
    const payload = safeParse(e.data);
    if (payload) opts.onFinalRow?.(payload as FinalRow);
  });

  es.addEventListener("done", (e: MessageEvent) => {
    const payload = safeParse(e.data);
    if (payload) opts.onDone?.(payload as StreamDonePayload);
    es.close();
  });

  // If server explicitly sends event:error with JSON payload
  es.addEventListener("error", (e: MessageEvent) => {
    const payload = safeParse((e as any).data ?? "");
    const msg =
      payload?.message ||
      "Streaming connection error. Check backend logs and CORS.";
    opts.onError?.(msg);
    es.close();
  });

  // Also handle generic errors (network disconnect, etc.)
  es.onerror = () => {
    opts.onError?.("Streaming connection lost. Is the backend running?");
    es.close();
  };

  return () => es.close();
}

/* =========================
   Downloads
========================= */

export function downloadCsv(
  runId: string,
  mode: "final" | "validated" | "proposals" | "coverage" = "final"
) {
  const url = `${API_BASE}/api/download.csv?run_id=${encodeURIComponent(
    runId
  )}&mode=${encodeURIComponent(mode)}`;
  window.open(url, "_blank");
}

/* =========================
   Suggestions
========================= */

export async function generateSuggestions(
  runId?: string | null,
  topN: number = 25
): Promise<SuggestionPayload> {
  const qs = new URLSearchParams();
  if (runId) qs.set("run_id", runId);
  qs.set("top_n", String(topN));

  const res = await fetch(`${API_BASE}/api/suggestions/generate?${qs.toString()}`, {
    method: "POST",
  });
  const text = await res.text();
  if (!res.ok) throw new Error(text || `Failed (${res.status})`);
  return JSON.parse(text) as SuggestionPayload;
}

export async function analyzeSuggestion(body: {
  segment_name: string;
  competitor_matches: SuggestionMatch[];
  cybba_matches: SuggestionMatch[];
  l1?: string;
  l2?: string;
}): Promise<SuggestionAnalysis> {
  const res = await fetch(`${API_BASE}/api/suggestions/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) throw new Error(text || `Failed (${res.status})`);
  return JSON.parse(text) as SuggestionAnalysis;
}