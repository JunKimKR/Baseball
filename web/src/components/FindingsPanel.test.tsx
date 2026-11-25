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

  const scoredOnly = (n: number) =>
    r.metrics.map((m, i) => ({ ...m, score: i < n ? 50 : null, z: i < n ? 0 : null }));

  it("shows disclaimer", () => {
    render(<FindingsPanel analysisId="id1" findings={r.findings} metrics={r.metrics} />);
    expect(screen.getByText("트레이닝 코치와 상의 후 적용하세요.")).toBeInTheDocument();
    expect(screen.getByText(r.findings[0].title)).toBeInTheDocument();
  });

  it("shows each cause's evidence source", () => {
    const f = r.findings.find((x) => x.causes.length > 0)!;
    render(<FindingsPanel analysisId="id1" findings={[f]} metrics={r.metrics} />);
    for (const c of f.causes) {
      expect(c.source.trim()).not.toBe("");
      expect(screen.getAllByText(c.source).length).toBeGreaterThan(0);
    }
    expect(screen.getAllByText("원인 근거").length).toBe(f.causes.length);
  });

  it("explains empty causes", () => {
    const f = r.findings.find((x) => x.causes.length === 0)!;
    render(<FindingsPanel analysisId="id1" findings={[f]} metrics={r.metrics} />);
    expect(screen.getByText(f.evidence)).toBeInTheDocument();
    expect(screen.getByText("근거가 확인된 원인 정보가 아직 없습니다.")).toBeInTheDocument();
  });

  it("narrows a cause only by its own self-test answer", async () => {
    const user = userEvent.setup();
    render(<FindingsPanel analysisId="id9" findings={[twoCause()]} metrics={r.metrics} />);
    // nothing chosen: every cause listed as possible, with exercises
    expect(screen.getAllByText("가능성")).toHaveLength(2);
    expect(screen.getByText("운동A")).toBeInTheDocument();
    expect(screen.getByText("운동B")).toBeInTheDocument();

    // A's test says A applies; B's test is unanswered, so B is not demoted
    await user.click(screen.getByLabelText("a 해당"));
    expect(screen.getByText("해당 가능성 높음")).toBeInTheDocument();
    expect(screen.getAllByText("가능성")).toHaveLength(1);
    expect(screen.getByText("운동A")).toBeInTheDocument();
    expect(screen.getByText("운동B")).toBeInTheDocument();
    expect(screen.queryByText(/가능성 낮음/)).not.toBeInTheDocument();
    expect(JSON.parse(localStorage.getItem("selftest:id9")!)).toEqual({ st_a: "yes" });

    // B's own answer says it does not apply: only now is it marked and folded
    await user.click(screen.getByLabelText("b 아님"));
    expect(screen.getByText("자가 테스트 결과상 덜 해당")).toBeInTheDocument();
    expect(screen.queryByText("운동B")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "운동 제안 펼치기" }));
    expect(screen.getByText("운동B")).toBeInTheDocument();
  });

  it("labels the self-test without a duration", () => {
    render(<FindingsPanel analysisId="id9" findings={[twoCause()]} metrics={r.metrics} />);
    expect(screen.getAllByText(/^자가 테스트:/).length).toBe(2);
    expect(screen.queryByText(/30초 자가 테스트/)).not.toBeInTheDocument();
  });

  it("restores answers from localStorage", () => {
    localStorage.setItem("selftest:id9", JSON.stringify({ st_b: "yes", st_a: "no" }));
    render(<FindingsPanel analysisId="id9" findings={[twoCause()]} metrics={r.metrics} />);
    expect(screen.getByText("운동B")).toBeInTheDocument();
    expect(screen.queryByText("운동A")).not.toBeInTheDocument();
    expect(screen.getByText("자가 테스트 결과상 덜 해당")).toBeInTheDocument();
  });

  it("shares one answer across findings with the same self-test, in separate radio groups", async () => {
    const user = userEvent.setup();
    const a = twoCause();
    const b = { ...twoCause(), issue_id: "y", title: "다른 문제" };
    render(<FindingsPanel analysisId="id7" findings={[a, b]} metrics={r.metrics} />);
    const radios = screen.getAllByLabelText("a 해당") as HTMLInputElement[];
    expect(radios).toHaveLength(2);
    expect(radios[0].name).not.toBe(radios[1].name);
    await user.click(radios[0]);
    expect(radios[0].checked).toBe(true);
    expect(radios[1].checked).toBe(true);
  });

  it("always says how many metrics could be evaluated", () => {
    const n = r.metrics.filter((m) => m.score !== null).length;
    render(<FindingsPanel analysisId="id1" findings={r.findings} metrics={r.metrics} />);
    expect(screen.getByText(`평가 가능한 지표 ${n}/${r.metrics.length}개`)).toBeInTheDocument();
  });

  it("does not call too few evaluated metrics 'no problems'", () => {
    render(<FindingsPanel analysisId="id1" findings={[]} metrics={scoredOnly(7)} />);
    expect(screen.getByText("평가 가능한 지표 7/15개")).toBeInTheDocument();
    expect(screen.getByText("평가된 지표가 부족해 진단하지 않았습니다.")).toBeInTheDocument();
    expect(screen.queryByText("뚜렷한 문제 지점이 발견되지 않았습니다.")).not.toBeInTheDocument();
  });

  it("says no clear problems only when enough metrics were evaluated", () => {
    render(<FindingsPanel analysisId="id1" findings={[]} metrics={scoredOnly(8)} />);
    expect(screen.getByText("뚜렷한 문제 지점이 발견되지 않았습니다.")).toBeInTheDocument();
    expect(screen.queryByText("평가된 지표가 부족해 진단하지 않았습니다.")).not.toBeInTheDocument();
  });
});
