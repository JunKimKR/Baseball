import { describe, expect, it } from "vitest";
import fixture from "../fixtures/sample-result.json";
import { activeLeaks, overlayConfidence, particlesFor, segmentStyle } from "./energy";
import type { AnalysisResult, Confidence, Leak, Metric } from "../types";

const r = fixture as unknown as AnalysisResult;

describe("overlayConfidence", () => {
  const withConf = (conf: Record<string, Confidence>): Metric[] =>
    r.metrics.map((m) => (m.name in conf ? { ...m, confidence: conf[m.name] } : m));

  it("takes each drawn column's confidence from the linear speed it shows", () => {
    const metrics = withConf({
      pelvis_peak_speed_ms: "high", shoulder_peak_speed_ms: "medium", wrist_peak_speed_ms: "low",
      pelvis_rotation_proxy_deg: "high", stride_length_pct_height: "low",
    });
    expect(overlayConfidence(metrics)).toEqual({ legs: "high", pelvis: "low", trunk: "medium", arm: "low" });
  });

  it("ignores segment confidence (it comes from other metrics)", () => {
    const camMoving = withConf({
      pelvis_peak_speed_ms: "low", shoulder_peak_speed_ms: "low", wrist_peak_speed_ms: "low",
    });
    expect(Object.values(overlayConfidence(camMoving))).toEqual(["low", "low", "low", "low"]);
  });

  it("is low when the metric is missing", () => {
    expect(overlayConfidence([])).toEqual({ legs: "low", pelvis: "low", trunk: "low", arm: "low" });
  });
});

describe("segmentStyle", () => {
  it("interpolates color and width by intensity", () => {
    const lo = segmentStyle(0, "high", false);
    const hi = segmentStyle(1, "high", false);
    expect(lo.color).toBe("#1e3a8a");
    expect(hi.color).toBe("#facc15");
    expect(lo.width).toBe(2);
    expect(hi.width).toBe(12);
    expect(lo.alpha).toBe(1);
    expect(lo.dash).toEqual([]);
  });
  it("low confidence dims", () => {
    const s = segmentStyle(0.5, "low", false);
    expect(s.alpha).toBe(0.35);
    expect(s.dash).toEqual([6, 4]);
  });
  it("low confidence never glows or adds up, and keeps its dashes visible", () => {
    // glow and additive particles would undo the 0.35 dimming of a strong but untrusted signal
    const low = segmentStyle(1, "low", false);
    expect(low.glow).toBe(false);
    expect(low.additive).toBe(false);
    expect(low.cap).toBe("butt"); // round caps fill 4 px dash gaps once the line is wide
    const high = segmentStyle(1, "medium", false);
    expect(high.glow).toBe(true);
    expect(high.additive).toBe(true);
    expect(high.cap).toBe("round");
    expect(segmentStyle(0.4, "high", false).glow).toBe(false);
    // a leak stays visible even on a low-confidence column
    const leak = segmentStyle(0.3, "low", true);
    expect(leak.glow).toBe(true);
    expect(leak.additive).toBe(true);
  });
  it("leaking is red", () => {
    expect(segmentStyle(0.5, "high", true).color).toBe("#ff3b30");
  });
});

describe("activeLeaks", () => {
  const leaks: Leak[] = [
    { segment: "arm", frame: 100, kind: "speed_drop", severity: 0.3, message: "" },
    { segment: "trunk", frame: 120, kind: "out_of_order", severity: 0.5, message: "" },
  ];
  it("keeps leaks within the window", () => {
    expect(activeLeaks(leaks, 103, 6)).toHaveLength(1);
    expect(activeLeaks(leaks, 106, 6)).toHaveLength(1);
    expect(activeLeaks(leaks, 107, 6)).toHaveLength(0);
    expect(activeLeaks([], 100, 6)).toEqual([]);
  });
});

describe("particlesFor", () => {
  const line: [[number, number], [number, number]][] = [[[0, 0], [100, 0]]];
  it("is deterministic for a fixed seed", () => {
    const a = particlesFor("arm", 0.5, 10, 7, { line });
    const b = particlesFor("arm", 0.5, 10, 7, { line });
    expect(a).toEqual(b);
  });
  it("count scales with intensity", () => {
    expect(particlesFor("arm", 0, 0, 1, { line })).toHaveLength(3);
    expect(particlesFor("arm", 1, 0, 1, { line })).toHaveLength(15);
    expect(particlesFor("arm", 0.5, 0, 1, { line }).length).toBeGreaterThan(3);
  });
  it("flows proximal to distal over time", () => {
    const x = (f: number) => particlesFor("arm", 0, f, 1, { line })[0].x;
    expect(x(0.1) - x(0)).toBeGreaterThan(0);
  });
  it("scatters off the segment when leaking", () => {
    const ps = particlesFor("arm", 1, 5, 1, { line, leaking: true });
    expect(ps.some((p) => Math.abs(p.y) > 1)).toBe(true);
    const ok = particlesFor("arm", 1, 5, 1, { line });
    expect(ok.every((p) => p.y === 0)).toBe(true);
  });
});
