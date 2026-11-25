import { describe, expect, it } from "vitest";
import fixture from "./fixtures/sample-result.json";
import type {
  AnalysisResult, AnalysisWarning, Cause, Energy, EnergyPeaks, Events, Exercise, Finding,
  Keypoints, Leak, Metric, Norm, Phase, SegmentScore, SelfTest, SelfTestOption, Subject, VideoMeta,
} from "./types";

/** Compile-time: the literal must list exactly every key of T (missing or extra -> tsc error). */
function keysOf<T>(o: Record<keyof T, true>): string[] {
  return Object.keys(o).sort();
}

const KEYS = {
  result: keysOf<AnalysisResult>({
    schema_version: true, meta: true, subject: true, warnings: true, throw_direction: true,
    keypoints: true, events: true, phases: true, metrics: true, segments: true, energy: true,
    findings: true,
  }),
  meta: keysOf<VideoMeta>({
    fps: true, width: true, height: true, n_frames: true, duration_s: true, capture_fps: true,
    norm_fps: true,
  }),
  subject: keysOf<Subject>({ height_cm: true, hand: true, camera_side: true }),
  warning: keysOf<AnalysisWarning>({ code: true, severity: true, message: true }),
  keypoints: keysOf<Keypoints>({ joints: true, frames: true }),
  events: keysOf<Events>({ leg_lift_peak: true, foot_plant: true, mer_est: true, release: true }),
  phase: keysOf<Phase>({ name: true, label: true, start: true, end: true }),
  metric: keysOf<Metric>({
    name: true, label: true, value: true, unit: true, confidence: true, segment: true,
    score: true, z: true, norm: true,
  }),
  norm: keysOf<Norm>({ mean: true, sd: true, direction: true, source: true }),
  segment: keysOf<SegmentScore>({ segment: true, label: true, score: true, confidence: true }),
  energy: keysOf<Energy>({ segments: true, intensity: true, leaks: true, peaks: true }),
  leak: keysOf<Leak>({ segment: true, frame: true, kind: true, severity: true, message: true }),
  peaks: keysOf<EnergyPeaks>({ pelvis_center: true, shoulder_center: true, elbow: true, wrist: true }),
  finding: keysOf<Finding>({ issue_id: true, title: true, evidence: true, causes: true }),
  cause: keysOf<Cause>({ id: true, title: true, source: true, self_test: true, exercises: true }),
  selfTest: keysOf<SelfTest>({
    id: true, instruction: true, options: true, implies_cause_on: true, source: true,
  }),
  option: keysOf<SelfTestOption>({ id: true, label: true }),
  exercise: keysOf<Exercise>({ id: true, name: true, sets_reps: true, description: true, source: true }),
};

function expectKeys(obj: unknown, expected: string[], where: string) {
  expect(obj, where).toBeTypeOf("object");
  expect(Object.keys(obj as object).sort(), where).toEqual(expected);
}

describe("sample-result fixture", () => {
  // JSON imports widen literals ("R" -> string): assertion, not assignment.
  const r = fixture as AnalysisResult;

  it("satisfies AnalysisResult shape at compile time", () => {
    expect(r.schema_version).toBe(1);
    expect(r.energy.intensity.length).toBe(r.meta.n_frames);
    expect(r.keypoints.frames.length).toBe(r.meta.n_frames);
  });

  it("has exactly the keys of every TS interface (drift guard)", () => {
    expectKeys(r, KEYS.result, "result");
    expectKeys(r.meta, KEYS.meta, "meta");
    expectKeys(r.subject, KEYS.subject, "subject");
    r.warnings.forEach((w, i) => expectKeys(w, KEYS.warning, `warnings[${i}]`));
    expectKeys(r.keypoints, KEYS.keypoints, "keypoints");
    expectKeys(r.events, KEYS.events, "events");
    r.phases.forEach((p, i) => expectKeys(p, KEYS.phase, `phases[${i}]`));
    r.metrics.forEach((m, i) => {
      expectKeys(m, KEYS.metric, `metrics[${i}]`);
      if (m.norm !== null) expectKeys(m.norm, KEYS.norm, `metrics[${i}].norm`);
    });
    r.segments.forEach((s, i) => expectKeys(s, KEYS.segment, `segments[${i}]`));
    expectKeys(r.energy, KEYS.energy, "energy");
    r.energy.leaks.forEach((l, i) => expectKeys(l, KEYS.leak, `energy.leaks[${i}]`));
    expectKeys(r.energy.peaks, KEYS.peaks, "energy.peaks");
    r.findings.forEach((f, i) => {
      expectKeys(f, KEYS.finding, `findings[${i}]`);
      f.causes.forEach((c, j) => {
        const w = `findings[${i}].causes[${j}]`;
        expectKeys(c, KEYS.cause, w);
        expectKeys(c.self_test, KEYS.selfTest, `${w}.self_test`);
        c.self_test.options.forEach((o, k) => expectKeys(o, KEYS.option, `${w}.self_test.options[${k}]`));
        c.exercises.forEach((e, k) => expectKeys(e, KEYS.exercise, `${w}.exercises[${k}]`));
      });
    });
  });

  it("covers the nested shapes it claims to (non-empty samples)", () => {
    expect(r.warnings.length).toBeGreaterThan(0);
    expect(r.phases.length).toBeGreaterThan(0);
    expect(r.metrics.some((m) => m.norm !== null)).toBe(true);
    expect(r.energy.leaks.length).toBeGreaterThan(0);
    expect(r.findings.some((f) => f.causes.some((c) => c.exercises.length > 0))).toBe(true);
  });
});
