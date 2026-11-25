import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import fixture from "../fixtures/sample-result.json";
import type { AnalysisResult, AnalysisState } from "../types";

type Poll = (id: string, onUpdate: (s: AnalysisState) => void, ms?: number, onError?: (e: Error) => void) => () => void;
let impl: Poll = () => () => {};
vi.mock("../api", async (orig) => ({
  ...(await orig<typeof import("../api")>()),
  pollAnalysis: (...a: Parameters<Poll>) => impl(...a),
  videoUrl: () => "/v.mp4",
}));
vi.mock("./VideoOverlay", () => ({ default: () => <div data-testid="overlay" /> }));

import AnalysisPage from "./AnalysisPage";
import { HttpError } from "../api";

const r = fixture as unknown as AnalysisResult;
const done = (result: AnalysisResult): AnalysisState => ({ status: "done", error: null, result });

describe("AnalysisPage", () => {
  beforeEach(() => {
    impl = () => () => {};
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
});
