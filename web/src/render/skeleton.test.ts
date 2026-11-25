import { describe, expect, it } from "vitest";
import sample from "../fixtures/sample-result.json";
import type { AnalysisResult } from "../types";
import { jointIndex, segmentJoints, segmentLines } from "./skeleton";

const r = sample as unknown as AnalysisResult;
const idx = jointIndex(r.keypoints.joints);

describe("skeleton", () => {
  it("arm follows the throwing hand", () => {
    expect(segmentJoints("R").arm[0][0]).toBe("right_shoulder");
    expect(segmentJoints("L").arm[0][0]).toBe("left_shoulder");
  });
  it("draws trunk lines from the fixture", () => {
    const lines = segmentLines(r.keypoints.frames[0], idx, segmentJoints("R").trunk);
    expect(lines).toHaveLength(2);
  });
  it("skips lines with null or low-visibility endpoints", () => {
    const f = r.keypoints.frames[0].map((p) => [...p]);
    f[idx.right_elbow] = [null, null, 0];
    f[idx.left_shoulder][2] = 0.2;
    expect(segmentLines(f, idx, segmentJoints("R").arm)).toHaveLength(0);
    expect(segmentLines(f, idx, segmentJoints("R").trunk)).toHaveLength(0);
  });
});
