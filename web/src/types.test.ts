import { describe, expect, it } from "vitest";
import fixture from "./fixtures/sample-result.json";
import type { AnalysisResult } from "./types";

describe("sample-result fixture", () => {
  it("satisfies AnalysisResult", () => {
    // JSON imports widen string literals ("R" -> string), so a plain assignment cannot work;
    // an assertion still fails to compile on missing keys or wrong structural types.
    const r = fixture as AnalysisResult;
    expect(Object.keys(r).sort()).toEqual(
      [
        "schema_version", "meta", "subject", "warnings", "throw_direction", "keypoints",
        "events", "phases", "metrics", "segments", "energy", "findings",
      ].sort(),
    );
    expect(r.schema_version).toBe(1);
    expect(Object.keys(r.energy.peaks).sort()).toEqual(["elbow", "pelvis_center", "shoulder_center", "wrist"]);
    expect(r.energy.intensity.length).toBe(r.meta.n_frames);
    expect(r.keypoints.frames.length).toBe(r.meta.n_frames);
  });
});
