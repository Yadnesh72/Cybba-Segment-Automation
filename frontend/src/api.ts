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
  validated: number;
  covered: number;

  duplicates_within_output_normalized?: number;

  dropped_naming: number;
  dropped_underived_only: number;
  dropped_net_new: number;
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

export interface RunResponse {
  run_id: string;
  summary: RunSummary;
  rows: ValidatedRow[];
}

/**
 * Option A backend emits:
 * - event: run_id  -> { run_id }
 * - event: summary -> RunSummary
 * - event: row     -> ValidatedRow
 * - event: done    -> { run_id, summary }
 * - event: error   -> { message, run_id? }
 */
export type StreamDonePayload = {
  run_id: string;
  summary: RunSummary;
};

export type StreamErrorPayload = {
  message: string;
  run_id?: string;
};

/* =========================
   API calls (non-stream)
========================= */

export async function runPipeline(): Promise<RunResponse> {
  const res = await fetch(`${API_BASE}/api/run`, { method: "POST" });

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
  onRunId?: (runId: string) => void;
  onSummary?: (summary: RunSummary) => void;
  onRow?: (row: ValidatedRow) => void;
  onDone?: (payload: StreamDonePayload) => void;
  onError?: (message: string, payload?: StreamErrorPayload) => void;
  onOpen?: () => void;
}): () => void {
  const url = `${API_BASE}/api/run/stream`;
  const es = new EventSource(url);

  const safeParse = (s: string) => {
    try {
      return JSON.parse(s);
    } catch {
      return null;
    }
  };

  es.onopen = () => {
    opts.onOpen?.();
  };

  // run_id arrives immediately (Option A)
  es.addEventListener("run_id", (e: MessageEvent) => {
    const payload = safeParse(e.data);
    if (payload?.run_id) {
      opts.onRunId?.(payload.run_id as string);
    }
  });

  es.addEventListener("summary", (e: MessageEvent) => {
    const payload = safeParse(e.data);
    if (payload) opts.onSummary?.(payload as RunSummary);
  });

  es.addEventListener("row", (e: MessageEvent) => {
    const payload = safeParse(e.data);
    if (payload) opts.onRow?.(payload as ValidatedRow);
  });

  // Option A done payload: { run_id, summary }
  es.addEventListener("done", (e: MessageEvent) => {
    const payload = safeParse(e.data);
    if (payload?.run_id) {
      opts.onDone?.(payload as StreamDonePayload);
    } else {
      // still close the stream; treat as done
      opts.onDone?.({ run_id: "", summary: (payload?.summary ?? {}) as RunSummary });
    }
    es.close();
  });

  // If server explicitly sends event:error with JSON payload
  es.addEventListener("error", (e: MessageEvent) => {
    const payload = safeParse((e as any).data ?? "");
    const msg =
      payload?.message ||
      "Streaming error event received. Check backend logs.";
    opts.onError?.(msg, payload as StreamErrorPayload);
    es.close();
  });

  // Network / CORS / disconnect error (EventSource-level)
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
