import { describe, expect, it, vi } from "vitest";
import { notifyOnChange, shortcutFor } from "./controls";

describe("shortcutFor", () => {
  const wrap = {};
  it("maps keys when the wrapper is the target", () => {
    expect(shortcutFor({ key: "ArrowLeft", target: wrap, currentTarget: wrap })).toBe("prev");
    expect(shortcutFor({ key: "ArrowRight", target: wrap, currentTarget: wrap })).toBe("next");
    expect(shortcutFor({ key: " ", target: wrap, currentTarget: wrap })).toBe("toggle");
    expect(shortcutFor({ key: "a", target: wrap, currentTarget: wrap })).toBeNull();
  });
  it("ignores events bubbling from buttons and inputs", () => {
    const button = document.createElement("button");
    const input = document.createElement("input");
    expect(shortcutFor({ key: " ", target: button, currentTarget: wrap })).toBeNull();
    expect(shortcutFor({ key: "ArrowRight", target: input, currentTarget: wrap })).toBeNull();
  });
});

describe("notifyOnChange", () => {
  it("fires only when the frame changes", () => {
    const cb = vi.fn();
    const n = notifyOnChange(cb);
    n(3); n(3); n(4); n(4); n(3);
    expect(cb.mock.calls.map((c) => c[0])).toEqual([3, 4, 3]);
  });
});
