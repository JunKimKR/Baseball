import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import fixture from "../fixtures/sample-result.json";
import type { AnalysisResult, AnalysisState } from "../types";

type Poll = (id: string, onUpdate: (s: AnalysisState) => void, ms?: number, onError?: (e: Error) => void) => () => void;
let impl: Poll = () => () => {};
vi.mock("../api", async (orig) => ({
  ...(await orig<typeof import("../api")>()),
  pollAnalysis: (...a: Parameters<Poll>) => impl(...a),
  videoUrl: () => "/v.mp4",
}));
type SeekTo = { frame: number; nonce: number } | undefined;
const overlay = vi.hoisted(() => ({ seekTo: [] as SeekTo[] }));
vi.mock("./VideoOverlay", () => ({
  default: (p: { seekTo?: SeekTo }) => {
    overlay.seekTo.push(p.seekTo);
    return <div data-testid="overlay" />;
  },
}));

import AnalysisPage from "./AnalysisPage";
import { HttpError } from "../api";

const r = fixture as unknown as AnalysisResult;
const done = (result: AnalysisResult): AnalysisState => ({ status: "done", error: null, result });

describe("AnalysisPage", () => {
  beforeEach(() => {
    impl = () => () => {};
    overlay.seekTo.length = 0;
  });

  it("says processing time depends on length and fps", () => {
    impl = (_id, onUpdate) => (onUpdate({ status: "processing", error: null, result: null }), () => {});
    render(<AnalysisPage id="x" />);
    expect(screen.getByRole("status")).toHaveTextContent("영상 길이와 fps에 따라 수십 초~수 분");
    expect(screen.getByRole("status")).not.toHaveTextContent("30초");
  });

  it("seeks again when the same phase is chosen twice", async () => {
    const user = userEvent.setup();
    impl = (_id, onUpdate) => (onUpdate(done(r)), () => {});
    render(<AnalysisPage id="x" />);
    const legLift = screen.getByRole("button", { name: /레그 리프트 프레임/ });
    await user.click(legLift);
    await user.click(legLift);
    const requests = [...new Set(overlay.seekTo.filter((t) => t !== undefined))];
    expect(requests).toHaveLength(2); // two distinct requests, so the overlay seeks twice
    expect(requests.map((t) => t!.frame)).toEqual([0, 0]);
    expect(requests[1]!.nonce).toBeGreaterThan(requests[0]!.nonce);
  });

  it("shows not-found instead of spinning", () => {
    impl = (_id, _u, _m, onError) => {
      onError?.(new HttpError("nf", 404));
      return () => {};
    };
    render(<AnalysisPage id="x" />);
    expect(screen.getByRole("alert")).toHaveTextContent("분석을 찾을 수 없습니다");
  });

  it("shows the server error for failed analyses", () => {
    impl = (_id, onUpdate) => {
      onUpdate({ status: "failed", error: "boom", result: null });
      return () => {};
    };
    render(<AnalysisPage id="x" />);
    expect(screen.getByRole("alert")).toHaveTextContent("boom");
  });

  it("error warning hides the overlay and shows the filming guide", () => {
    const bad = { ...r, warnings: [{ code: "NO_PERSON", severity: "error" as const, message: "투수를 인식하지 못했습니다." }] };
    impl = (_id, onUpdate) => (onUpdate(done(bad)), () => {});
    render(<AnalysisPage id="x" />);
    expect(screen.queryByTestId("overlay")).not.toBeInTheDocument();
    expect(screen.getByText("투수를 인식하지 못했습니다.")).toBeInTheDocument();
    expect(screen.getByText("촬영 가이드")).toBeInTheDocument();
    expect(screen.getByText("트레이닝 코치와 상의 후 적용하세요.")).toBeInTheDocument();
  });

  it("renders repeated warning codes without key collisions", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const w = (message: string) => ({ code: "EVENT_NOT_FOUND", severity: "warn" as const, message });
    const twice = { ...r, warnings: [w("앞발 착지 없음"), w("릴리스 없음")] };
    impl = (_id, onUpdate) => (onUpdate(done(twice)), () => {});
    render(<AnalysisPage id="x" />);
    expect(screen.getByText("앞발 착지 없음")).toBeInTheDocument();
    expect(screen.getByText("릴리스 없음")).toBeInTheDocument();
    expect(spy.mock.calls.some((c) => String(c[0]).includes("same key"))).toBe(false);
    spy.mockRestore();
  });

  it("lists errors first, then warnings, then info", () => {
    const w = (code: string, severity: "info" | "warn" | "error") => ({ code, severity, message: `m-${code}` });
    const mixed = { ...r, warnings: [w("I", "info"), w("W", "warn"), w("E", "error")] };
    impl = (_id, onUpdate) => (onUpdate(done(mixed)), () => {});
    render(<AnalysisPage id="x" />);
    const items = screen.getAllByText(/^m-/).map((e) => /m-\w/.exec(e.textContent ?? "")![0]);
    expect(items).toEqual(["m-E", "m-W", "m-I"]);
  });

  it("renders the full page with reference-only scores", () => {
    impl = (_id, onUpdate) => (onUpdate(done(r)), () => {});
    render(<AnalysisPage id="x" />);
    expect(screen.getByTestId("overlay")).toBeInTheDocument();
    expect(screen.getByText("부위별 점수")).toBeInTheDocument();
    expect(screen.getAllByText("참고값").length).toBeGreaterThan(0);
    expect(screen.getByText("트레이닝 코치와 상의 후 적용하세요.")).toBeInTheDocument();
  });

  it("shows the container, analysis and norm frame rates", () => {
    impl = (_id, onUpdate) => (onUpdate(done(r)), () => {});
    render(<AnalysisPage id="x" />);
    expect(screen.getByText("영상 60fps · 분석 60fps · 비교 기준값 60fps")).toBeInTheDocument();
  });

  it("converts frames to ms with the capture fps of a slow-motion export", () => {
    // same frames, container says 30: timing texts must not change (analysis fps is still 60)
    impl = (_id, onUpdate) => (onUpdate(done(r)), () => {});
    const { unmount } = render(<AnalysisPage id="x" />);
    const timing = () => screen.getAllByText(/최고속도 시점: 기준보다/).map((e) => e.textContent);
    const before = timing();
    unmount();
    const slowmo = { ...r, meta: { ...r.meta, fps: 30, capture_fps: 60 } };
    impl = (_id, onUpdate) => (onUpdate(done(slowmo)), () => {});
    render(<AnalysisPage id="x" />);
    expect(screen.getByText("영상 30fps · 분석 60fps · 비교 기준값 60fps")).toBeInTheDocument();
    expect(timing()).toEqual(before);
  });

  it("marks phases without scored metrics as not evaluated", () => {
    impl = (_id, onUpdate) => (onUpdate(done(r)), () => {});
    render(<AnalysisPage id="x" />);
    const legLift = screen.getByRole("button", { name: /레그 리프트 프레임/ });
    expect(legLift).toHaveTextContent("평가 안 됨");
    expect(legLift).not.toHaveTextContent("양호");
    expect(screen.getByRole("button", { name: /레그 리프트 구간으로 이동 — 평가 안 됨/ })).toBeInTheDocument();
  });
});
