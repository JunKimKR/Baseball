import type { Confidence, Leak, Metric } from "../types";
import type { Pt, Segment } from "./skeleton";

export const LEAK_COLOR = "#ff3b30";
const LOW: [number, number, number] = [0x1e, 0x3a, 0x8a];
const HIGH: [number, number, number] = [0xfa, 0xcc, 0x15];

export interface SegmentStyle {
  color: string;
  width: number;
  alpha: number;
  dash: number[];
  /** Halo around the line; additive blending of line and particles. Off for low confidence
   * (unless leaking): both brighten a strong but untrusted signal past its 0.35 dimming. */
  glow: boolean;
  additive: boolean;
  /** Butt caps keep a low-confidence line's dashes visible; round caps fill the gaps when wide. */
  cap: CanvasLineCap;
}

const clamp01 = (v: number) => Math.min(Math.max(Number.isFinite(v) ? v : 0, 0), 1);
const hex = (n: number) => Math.round(n).toString(16).padStart(2, "0");

export function energyColor(intensity: number): string {
  const t = clamp01(intensity);
  return "#" + LOW.map((lo, i) => hex(lo + (HIGH[i] - lo) * t)).join("");
}

export function segmentStyle(intensity: number, confidence: Confidence, leaking: boolean): SegmentStyle {
  const t = clamp01(intensity);
  const low = confidence === "low";
  return {
    color: leaking ? LEAK_COLOR : energyColor(t),
    width: 2 + 10 * t,
    alpha: low ? 0.35 : 1,
    dash: low ? [6, 4] : [],
    glow: leaking || (!low && t > 0.5),
    additive: leaking || !low,
    cap: low ? "butt" : "round",
  };
}

/**
 * The metric behind each drawn energy column: the overlay shows these linear speeds (energy.py
 * normalizes the same signals), so their confidence, not the segment score's, sets the
 * dimming. The pelvis column is a rotation proxy and always low.
 */
export const OVERLAY_SIGNAL: Record<Segment, string | null> = {
  legs: "pelvis_peak_speed_ms",
  pelvis: null,
  trunk: "shoulder_peak_speed_ms",
  arm: "wrist_peak_speed_ms",
};

/** Confidence per drawn column; a missing metric counts as low. */
export function overlayConfidence(metrics: Metric[]): Record<Segment, Confidence> {
  const byName = new Map(metrics.map((m) => [m.name, m.confidence]));
  const at = (name: string | null): Confidence => (name === null ? "low" : byName.get(name) ?? "low");
  return {
    legs: at(OVERLAY_SIGNAL.legs),
    pelvis: at(OVERLAY_SIGNAL.pelvis),
    trunk: at(OVERLAY_SIGNAL.trunk),
    arm: at(OVERLAY_SIGNAL.arm),
  };
}

/** Leaks within ±windowFrames of the frame. */
export function activeLeaks(leaks: Leak[], frame: number, windowFrames: number): Leak[] {
  return leaks.filter((l) => Math.abs(l.frame - frame) <= windowFrames);
}

export interface Particle {
  x: number;
  y: number;
  r: number;
  alpha: number;
}

export interface ParticleOpts {
  /** Polyline pieces of the segment, proximal → distal (any coordinate space). */
  line?: [Pt, Pt][];
  leaking?: boolean;
}

/** Small deterministic hash → [0, 1). */
function rand(a: number, b: number): number {
  let h = (Math.imul(a | 0, 374761393) + Math.imul(b | 0, 668265263)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

const SEG_SALT: Record<string, number> = { legs: 1, pelvis: 2, trunk: 3, arm: 4 };

/**
 * Particles streaming proximal → distal along the segment; `frame` may be fractional for
 * smooth motion. When leaking they drift off the segment instead of staying on it.
 */
export function particlesFor(
  segment: Segment | string,
  intensity: number,
  frame: number,
  seed: number,
  opts: ParticleOpts = {},
): Particle[] {
  const line = opts.line ?? [[[0, 0], [1, 0]] as [Pt, Pt]];
  const t = clamp01(intensity);
  const n = Math.round(3 + 12 * t);
  const lens = line.map(([a, b]) => Math.hypot(b[0] - a[0], b[1] - a[1]));
  const total = lens.reduce((s, l) => s + l, 0);
  const salt = (SEG_SALT[segment] ?? 9) * 1000 + seed;
  const speed = 0.02 + 0.03 * t; // fraction of the segment per frame
  const out: Particle[] = [];
  for (let i = 0; i < n; i++) {
    const phase = i / n + rand(salt, i) * 0.5 / n;
    const u = (((phase + frame * speed) % 1) + 1) % 1;
    // locate u along the polyline
    let d = u * total;
    let k = 0;
    while (k < line.length - 1 && d > lens[k]) {
      d -= lens[k];
      k++;
    }
    const [a, b] = line[k];
    const f = lens[k] > 0 ? d / lens[k] : 0;
    let x = a[0] + (b[0] - a[0]) * f;
    let y = a[1] + (b[1] - a[1]) * f;
    let alpha = 0.5 + 0.5 * t;
    let r = 1.5 + 2.5 * t;
    if (opts.leaking) {
      // scatter increases toward the distal end; direction is fixed per particle
      const ang = rand(salt + 77, i) * Math.PI * 2;
      const spread = (0.15 + u) * (0.12 * (total || 1)) * (0.5 + rand(salt + 91, i));
      x += Math.cos(ang) * spread;
      y += Math.sin(ang) * spread;
      alpha *= 1 - 0.6 * u;
      r *= 0.8;
    }
    out.push({ x, y, r, alpha });
  }
  return out;
}
