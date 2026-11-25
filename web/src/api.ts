import type { AnalysisState, AnalysisSummary, CameraSide, Hand } from "./types";

export interface UploadOptions {
  heightCm: number;
  hand: Hand;
  cameraSide: CameraSide;
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

export async function uploadVideo(file: File, opts: UploadOptions): Promise<string> {
  const fd = new FormData();
  fd.append("video", file);
  fd.append("height_cm", String(opts.heightCm));
  fd.append("hand", opts.hand);
  fd.append("camera_side", opts.cameraSide);
  const res = await fetch("/api/analyses", { method: "POST", body: fd });
  if (!res.ok) throw new Error(await errorMessage(res));
  const body = await res.json();
  return body.id as string;
}

export async function getAnalysis(id: string): Promise<AnalysisState> {
  const res = await fetch(`/api/analyses/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(await errorMessage(res));
  const b = await res.json();
  return { status: b.status, error: b.error ?? null, result: b.result ?? null };
}

/** Polls until done/failed; returns a cancel function. Network errors are retried. */
export function pollAnalysis(
  id: string,
  onUpdate: (state: AnalysisState) => void,
  intervalMs = 1000,
): () => void {
  let cancelled = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const tick = async () => {
    let next = true;
    try {
      const state = await getAnalysis(id);
      if (cancelled) return;
      onUpdate(state);
      if (state.status !== "processing") next = false;
    } catch {
      if (cancelled) return;
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
