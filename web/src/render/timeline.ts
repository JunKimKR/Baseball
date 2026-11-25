import type { EnergyPeaks, Leak, Metric, Phase } from "../types";

export interface PhaseRect {
  name: string;
  label: string;
  x: number;
  w: number;
}

/** Phase bars laid out proportionally over `widthPx` (frames 0..nFrames-1). */
export function phaseRects(phases: Phase[], nFrames: number, widthPx: number): PhaseRect[] {
  const span = Math.max(nFrames - 1, 1);
  return phases.map((p) => ({
    name: p.name,
    label: p.label,
    x: (p.start / span) * widthPx,
    w: ((p.end - p.start) / span) * widthPx,
  }));
}

export interface GhostMarker {
  label: string;
  actualFrame: number;
  idealFrame: number;
}

const GHOST_CHAIN: Array<{ label: string; gap: string; peak: keyof EnergyPeaks }> = [
  { label: "어깨", gap: "gap_pelvis_shoulder_ms", peak: "shoulder_center" },
  { label: "팔꿈치", gap: "gap_shoulder_elbow_ms", peak: "elbow" },
  { label: "손목", gap: "gap_elbow_wrist_ms", peak: "wrist" },
];

/**
 * Timing ghost: the ideal peak frames, accumulated from the ACTUAL pelvis-centre
 * peak by the reference mean of each gap metric. Stops at the first gap whose
 * value or norm is missing.
 */
export function ghostMarkers(metrics: Metric[], peaks: EnergyPeaks, fps: number): GhostMarker[] {
  if (peaks.pelvis_center === null) return [];
  const out: GhostMarker[] = [];
  let ideal = peaks.pelvis_center;
  for (const step of GHOST_CHAIN) {
    const m = metrics.find((x) => x.name === step.gap);
    if (!m || m.value === null || m.norm === null) break;
    ideal += (m.norm.mean / 1000) * fps;
    const actual = peaks[step.peak];
    if (actual === null) continue;
    out.push({ label: step.label, actualFrame: actual, idealFrame: Math.round(ideal) });
  }
  return out;
}

export const METRIC_PHASE: Record<string, string> = {
  stride_length_pct_height: "stride",
  lead_knee_flexion_fp_deg: "arm_cocking",
  lead_knee_extension_deg: "arm_cocking",
  trunk_flexion_br_deg: "acceleration",
  wrist_peak_speed_ms: "acceleration",
  elbow_ext_peak_vel_dps: "acceleration",
  gap_elbow_wrist_ms: "acceleration",
  gap_pelvis_shoulder_ms: "arm_cocking",
  gap_shoulder_elbow_ms: "arm_cocking",
  shoulder_peak_speed_ms: "arm_cocking",
  pelvis_peak_speed_ms: "arm_cocking",
};

export const LOW_SCORE = 30;

export function phaseStatus(phase: Phase, metrics: Metric[], leaks: Leak[]): "ok" | "warn" {
  if (leaks.some((l) => l.frame >= phase.start && l.frame <= phase.end)) return "warn";
  const low = metrics.some(
    (m) => METRIC_PHASE[m.name] === phase.name && m.score !== null && m.score < LOW_SCORE,
  );
  return low ? "warn" : "ok";
}

export const CONFIDENCE_LABEL: Record<string, string> = { high: "높음", medium: "중간", low: "낮음" };

/** Short reason shown next to a "—" value; the warning list explains the common cases. */
export function missingReason(m: Metric, warnings: Array<{ code: string }>): string {
  const has = (c: string) => warnings.some((w) => w.code === c);
  if (m.unit === "ms" && has("TIMING_DISABLED")) return "프레임 레이트 부족으로 타이밍 지표 비활성";
  if (has("KEYPOINTS_MISSING")) return "관절이 인식되지 않은 구간이 있음";
  return "측정되지 않음";
}
