import { describe, expect, it } from "vitest";
import { frameAt, toCanvas } from "./frame";

describe("frameAt", () => {
  it("rounds and clamps", () => {
    expect(frameAt(1.0, 30, 100)).toBe(30);
    expect(frameAt(0.0166, 30, 100)).toBe(0);
    expect(frameAt(0.0174, 30, 100)).toBe(1);
    expect(frameAt(-1, 30, 100)).toBe(0);
    expect(frameAt(99, 30, 100)).toBe(99);
  });
});

describe("toCanvas", () => {
  it("letterboxes a portrait video in a landscape canvas", () => {
    const video = { w: 1080, h: 1920 };
    const canvas = { w: 800, h: 600 };
    // scale = 600/1920 = 0.3125; drawn width 337.5; offset (800-337.5)/2 = 231.25
    const [x0, y0] = toCanvas(0, 0, video, canvas);
    expect(x0).toBeCloseTo(231.25);
    expect(y0).toBeCloseTo(0);
    const [x1, y1] = toCanvas(1080, 1920, video, canvas);
    expect(x1).toBeCloseTo(568.75);
    expect(y1).toBeCloseTo(600);
  });
  it("pillarboxes a wide video in a tall canvas", () => {
    const [x, y] = toCanvas(0, 0, { w: 1920, h: 1080 }, { w: 960, h: 960 });
    expect(x).toBeCloseTo(0);
    expect(y).toBeCloseTo((960 - 540) / 2);
  });
});
