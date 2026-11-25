import { useEffect, useState } from "react";
import { pollAnalysis } from "../api";
import type { AnalysisState } from "../types";

/** Placeholder until the analysis page (Task 13). */
export default function AnalysisPlaceholder({ id }: { id: string }) {
  const [state, setState] = useState<AnalysisState | null>(null);
  useEffect(() => {
    setState(null);
    return pollAnalysis(id, setState);
  }, [id]);

  const label =
    state === null
      ? "불러오는 중…"
      : state.status === "processing"
        ? "분석 중입니다…"
        : state.status === "done"
          ? "분석이 완료되었습니다."
          : `분석에 실패했습니다${state.error ? `: ${state.error}` : ""}`;
  return (
    <section className="card">
      <a href="#/" className="back">← 처음으로</a>
      <h1>분석 결과</h1>
      <p className={state?.status === "failed" ? "error" : "muted"} role="status">{label}</p>
    </section>
  );
}
