import { beforeEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import fixture from "../fixtures/sample-result.json";
import type { AnalysisResult, Finding } from "../types";
import FindingsPanel from "./FindingsPanel";

const r = fixture as unknown as AnalysisResult;

const twoCause = (): Finding => {
  const base = r.findings.find((f) => f.causes.length > 0)!.causes[0];
  const mk = (id: string, title: string, exName: string) => ({
    ...base,
    id,
    title,
    self_test: {
      ...base.self_test,
      id: `st_${id}`,
      options: [
        { id: "no", label: `${id} 아님` },
        { id: "yes", label: `${id} 해당` },
      ],
      implies_cause_on: "yes",
    },
    exercises: [{ ...base.exercises[0], id: `ex_${id}`, name: exName }],
  });
  return {
    issue_id: "x", title: "테스트 문제", evidence: "근거 문장",
    causes: [mk("a", "원인 A", "운동A"), mk("b", "원인 B", "운동B")],
  };
};

describe("FindingsPanel", () => {
  beforeEach(() => localStorage.clear());

  it("shows disclaimer", () => {
    render(<FindingsPanel analysisId="id1" findings={r.findings} />);
    expect(screen.getByText("트레이닝 코치와 상의 후 적용하세요.")).toBeInTheDocument();
    expect(screen.getByText(r.findings[0].title)).toBeInTheDocument();
  });

  it("explains empty causes", () => {
    const f = r.findings.find((x) => x.causes.length === 0)!;
    render(<FindingsPanel analysisId="id1" findings={[f]} />);
    expect(screen.getByText(f.evidence)).toBeInTheDocument();
    expect(screen.getByText("근거가 확인된 원인 정보가 아직 없습니다.")).toBeInTheDocument();
  });

  it("narrows causes after self-test", async () => {
    const user = userEvent.setup();
    render(<FindingsPanel analysisId="id9" findings={[twoCause()]} />);
    // nothing chosen: every cause listed as possible, with exercises
    expect(screen.getAllByText("가능성")).toHaveLength(2);
    expect(screen.getByText("운동A")).toBeInTheDocument();
    expect(screen.getByText("운동B")).toBeInTheDocument();

    await user.click(screen.getByLabelText("a 해당"));
    expect(screen.getByText("운동A")).toBeInTheDocument();
    expect(screen.queryByText("운동B")).not.toBeInTheDocument();
    expect(JSON.parse(localStorage.getItem("selftest:id9")!)).toEqual({ st_a: "yes" });
  });

  it("restores answers from localStorage", () => {
    localStorage.setItem("selftest:id9", JSON.stringify({ st_b: "yes" }));
    render(<FindingsPanel analysisId="id9" findings={[twoCause()]} />);
    expect(screen.getByText("운동B")).toBeInTheDocument();
    expect(screen.queryByText("운동A")).not.toBeInTheDocument();
  });

  it("shares one answer across findings with the same self-test, in separate radio groups", async () => {
    const user = userEvent.setup();
    const a = twoCause();
    const b = { ...twoCause(), issue_id: "y", title: "다른 문제" };
    render(<FindingsPanel analysisId="id7" findings={[a, b]} />);
    const radios = screen.getAllByLabelText("a 해당") as HTMLInputElement[];
    expect(radios).toHaveLength(2);
    expect(radios[0].name).not.toBe(radios[1].name);
    await user.click(radios[0]);
    expect(radios[0].checked).toBe(true);
    expect(radios[1].checked).toBe(true);
  });
});
