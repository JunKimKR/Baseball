import { describe, expect, it } from "vitest";
import fixture from "../fixtures/sample-result.json";
import type { AnalysisResult, Leak, Metric } from "../types";
import { PHASE_STATUS_LABEL, ghostMarkers, phaseRects, phaseStatus } from "./timeline";

const r = fixture as unknown as AnalysisResult;
const phase = (name: string) => r.phases.find((p) => p.name === name)!;

describe("timeline", () => {
  it("phaseRects proportional widths", () => {
    const phases = [
      { name: "a", label: "A", start: 0, end: 50 },
      { name: "b", label: "B", start: 50, end: 100 },
    ];
    const rects = phaseRects(phases, 101, 200);
    expect(rects[0]).toMatchObject({ name: "a", label: "A", x: 0, w: 100 });
    expect(rects[1].x).toBeCloseTo(100);
    expect(rects[1].w).toBeCloseTo(100);
  });

  it("ghostMarkers cumulative from pelvis peak", () => {
    const peaks = { pelvis_center: 100, shoulder_center: 110, elbow: 125, wrist: 130 };
    const norm = (mean: number) => ({ mean, sd: 1, direction: "target" as const, source: "s" });
    const m = (name: string, mean: number): Metric => ({
      name, label: name, value: 10, unit: "ms", confidence: "high", segment: "arm",
      score: 50, z: 0, norm: norm(mean),
    });
    const metrics = [m("gap_pelvis_shoulder_ms", 50), m("gap_shoulder_elbow_ms", 100), m("gap_elbow_wrist_ms", 25)];
    const g = ghostMarkers(metrics, peaks, 100); // 100 fps: 1 frame = 10 ms
    expect(g.map((x) => x.idealFrame)).toEqual([105, 115, 118]); // 100+5, +10, +2.5 -> round
    expect(g.map((x) => x.actualFrame)).toEqual([110, 125, 130]);
    expect(g.map((x) => x.label)).toEqual(["어깨", "팔꿈치", "손목"]);
  });

  it("ghostMarkers skips null", () => {
    const real = ghostMarkers(r.metrics, r.energy.peaks, r.meta.fps);
    expect(real).toHaveLength(3);
    // missing pelvis peak -> nothing
    expect(ghostMarkers(r.metrics, { ...r.energy.peaks, pelvis_center: null }, r.meta.fps)).toEqual([]);
    // null value in the 2nd gap drops it and everything after
    const metrics = r.metrics.map((x) => (x.name === "gap_shoulder_elbow_ms" ? { ...x, value: null } : x));
    expect(ghostMarkers(metrics, r.energy.peaks, r.meta.fps)).toHaveLength(1);
    const noNorm = r.metrics.map((x) => (x.name === "gap_pelvis_shoulder_ms" ? { ...x, norm: null } : x));
    expect(ghostMarkers(noNorm, r.energy.peaks, r.meta.fps)).toEqual([]);
  });

  it("phaseStatus warns on leak or low score", () => {
    const noLeaks: Leak[] = [];
    // fixture: gap_elbow_wrist score 0 -> acceleration warns; stride_length 83 -> stride ok
    expect(phaseStatus(phase("acceleration"), r.metrics, noLeaks)).toBe("warn");
    expect(phaseStatus(phase("stride"), r.metrics, noLeaks)).toBe("ok");
    // lead_knee_flexion score 0 -> arm_cocking warns
    expect(phaseStatus(phase("arm_cocking"), r.metrics, noLeaks)).toBe("warn");
    // a leak inside a phase warns even if scores are fine
    const stride = phase("stride");
    const mid = Math.round((stride.start + stride.end) / 2);
    const leak: Leak = { segment: "arm", frame: mid, kind: "speed_drop", severity: 0.4, message: "x" };
    expect(phaseStatus(phase("stride"), r.metrics, [leak])).toBe("warn");
    // fp_to_br and trunk_flexion_peak_vel also belong to acceleration
    for (const name of ["fp_to_br_ms", "trunk_flexion_peak_vel_dps"]) {
      const ms = r.metrics.map((x) => ({ ...x, score: x.name === name ? 10 : 100 }));
      expect(phaseStatus(phase("acceleration"), ms, noLeaks)).toBe("warn");
    }
  });

  it("phaseStatus is unevaluated, not ok, when nothing in the phase was scored", () => {
    const noLeaks: Leak[] = [];
    // no metric maps to leg lift or follow-through
    expect(phaseStatus(phase("leg_lift"), r.metrics, noLeaks)).toBe("unevaluated");
    expect(phaseStatus(phase("follow_through"), r.metrics, noLeaks)).toBe("unevaluated");
    // mapped metrics without a score (e.g. 30 fps, camera motion) are not evidence of "ok"
    const nulls = r.metrics.map((x) => ({ ...x, score: null }));
    expect(phaseStatus(phase("acceleration"), nulls, noLeaks)).toBe("unevaluated");
    // ... but a leak in the phase still warns
    const ft = phase("follow_through");
    const leak: Leak = { segment: "arm", frame: ft.start + 1, kind: "speed_drop", severity: 0.4, message: "x" };
    expect(phaseStatus(ft, nulls, [leak])).toBe("warn");
    expect(PHASE_STATUS_LABEL).toEqual({ ok: "양호", warn: "점검 필요", unevaluated: "평가 안 됨" });
  });
});

describe("missingReason", () => {
  it("explains ms metrics when timing is disabled", async () => {
    const { missingReason } = await import("./timeline");
    const ms = r.metrics.find((m) => m.unit === "ms")!;
    expect(missingReason(ms, [{ code: "TIMING_DISABLED" }])).toContain("프레임 레이트");
    expect(missingReason(ms, [])).toBe("측정되지 않음");
  });
});
