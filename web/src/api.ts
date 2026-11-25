import type { AnalysisState, AnalysisSummary, CameraSide, Hand } from "./types";

export interface UploadOptions {
  heightCm: number;
  hand: Hand;
  cameraSide: CameraSide;
  /** Frame rate the clip was shot at, when the file says otherwise (slow-motion export). */
  captureFps?: number;
}

/** Turn a FastAPI error body (`detail` string or validation list) into a message. */
async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    const d = body?.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d.map((e) => (typeof e?.msg === "string" ? e.msg : JSON.stringify(e))).join("; ");
    }
  } catch {
    /* not JSON */
  }
  return `요청 실패 (${res.status})`;
}

/** Error from a non-2xx response; `status` lets callers tell "not found" from transient failures. */
export class HttpError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "HttpError";
  }
}

export async function uploadVideo(file: File, opts: UploadOptions): Promise<string> {
  const fd = new FormData();
  fd.append("video", file);
  fd.append("height_cm", String(opts.heightCm));
  fd.append("hand", opts.hand);
  fd.append("camera_side", opts.cameraSide);
  if (opts.captureFps !== undefined) fd.append("capture_fps", String(opts.captureFps));
  const res = await fetch("/api/analyses", { method: "POST", body: fd });
  if (!res.ok) throw new Error(await errorMessage(res));
  const body = await res.json();
  return body.id as string;
}

export async function getAnalysis(id: string): Promise<AnalysisState> {
  const res = await fetch(`/api/analyses/${encodeURIComponent(id)}`);
  if (!res.ok) throw new HttpError(await errorMessage(res), res.status);
  const b = await res.json();
  return { status: b.status, error: b.error ?? null, result: b.result ?? null };
}

/** After this many consecutive failures polling gives up and reports through `onError`. */
export const MAX_POLL_FAILURES = 5;

/**
 * Polls until done/failed; returns a cancel function. Transient errors are retried,
 * but a 404 or MAX_POLL_FAILURES consecutive failures stop polling and call `onError`.
 */
export function pollAnalysis(
  id: string,
  onUpdate: (state: AnalysisState) => void,
  intervalMs = 1000,
  onError?: (err: Error) => void,
): () => void {
  let cancelled = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let failures = 0;

  const tick = async () => {
    let next = true;
    try {
      const state = await getAnalysis(id);
      if (cancelled) return;
      failures = 0;
      onUpdate(state);
      if (state.status !== "processing") next = false;
    } catch (err) {
      if (cancelled) return;
      failures += 1;
      const notFound = err instanceof HttpError && err.status === 404;
      if (notFound || failures >= MAX_POLL_FAILURES) {
        onError?.(err instanceof Error ? err : new Error(String(err)));
        next = false;
      }
    }
    if (next && !cancelled) timer = setTimeout(tick, intervalMs);
  };
  void tick();

  return () => {
    cancelled = true;
    if (timer !== undefined) clearTimeout(timer);
  };
}

export function videoUrl(id: string): string {
  return `/api/analyses/${encodeURIComponent(id)}/video`;
}

export async function listAnalyses(): Promise<AnalysisSummary[]> {
  const res = await fetch("/api/analyses");
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}
