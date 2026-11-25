import type { Hand, KeypointFrame, SegmentName } from "../types";

export type Segment = SegmentName;
export type Pt = [number, number];

/** Virtual joints computed as the midpoint of two real ones. */
export const CENTERS: Record<string, [string, string]> = {
  hip_center: ["left_hip", "right_hip"],
  shoulder_center: ["left_shoulder", "right_shoulder"],
};

/** Joint pairs per segment for a right-handed thrower (arm uses right_*). */
export const SEGMENT_JOINTS: Record<Segment, [string, string][]> = segmentJoints("R");

/** Joint pairs per segment; the arm follows the throwing hand. */
export function segmentJoints(hand: Hand): Record<Segment, [string, string][]> {
  const t = hand === "R" ? "right" : "left";
  const legs: [string, string][] = [];
  for (const side of ["left", "right"]) {
    legs.push(
      [`${side}_hip`, `${side}_knee`],
      [`${side}_knee`, `${side}_ankle`],
      [`${side}_ankle`, `${side}_foot_index`],
    );
  }
  return {
    legs,
    pelvis: [["left_hip", "right_hip"]],
    trunk: [
      ["hip_center", "shoulder_center"],
      ["left_shoulder", "right_shoulder"],
    ],
    arm: [
      [`${t}_shoulder`, `${t}_elbow`],
      [`${t}_elbow`, `${t}_wrist`],
    ],
  };
}

export const MIN_VISIBILITY = 0.5;

function joint(frame: KeypointFrame, idx: Record<string, number>, name: string): Pt | null {
  const i = idx[name];
  const p = i === undefined ? undefined : frame[i];
  if (!p) return null;
  const [x, y, v] = p;
  if (x == null || y == null || v == null || v < MIN_VISIBILITY) return null;
  return [x, y];
}

/** Position of a joint (or virtual center) in video pixels; null if missing/low visibility. */
export function jointPoint(frame: KeypointFrame, idx: Record<string, number>, name: string): Pt | null {
  const c = CENTERS[name];
  if (c) {
    const a = joint(frame, idx, c[0]);
    const b = joint(frame, idx, c[1]);
    return a && b ? [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] : null;
  }
  return joint(frame, idx, name);
}

export function jointIndex(joints: string[]): Record<string, number> {
  return Object.fromEntries(joints.map((n, i) => [n, i]));
}

/** Drawable line segments (video pixels) for one segment; pairs with a missing end are skipped. */
export function segmentLines(
  frame: KeypointFrame,
  idx: Record<string, number>,
  pairs: [string, string][],
): [Pt, Pt][] {
  const out: [Pt, Pt][] = [];
  for (const [a, b] of pairs) {
    const pa = jointPoint(frame, idx, a);
    const pb = jointPoint(frame, idx, b);
    if (pa && pb) out.push([pa, pb]);
  }
  return out;
}
