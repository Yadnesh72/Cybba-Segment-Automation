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

  const url =
    maxRows !== undefined
      ? `${API_BASE}/api/run/stream?max_rows=${encodeURIComponent(maxRows)}`
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
