import { useEffect, useState } from "react";
import type { Cause, Finding, Metric } from "../types";

/** With no findings and fewer scored metrics than this, "no problems" would claim too much. */
export const MIN_SCORED_FOR_DIAGNOSIS = 8;

type Answers = Record<string, string>;

export function loadAnswers(analysisId: string): Answers {
  try {
    const raw = localStorage.getItem(`selftest:${analysisId}`);
    const v = raw ? JSON.parse(raw) : {};
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Answers) : {};
  } catch {
    return {};
  }
}

function saveAnswers(analysisId: string, answers: Answers): void {
  try {
    localStorage.setItem(`selftest:${analysisId}`, JSON.stringify(answers));
  } catch {
    /* storage unavailable: answers stay in memory only */
  }
}

function Source({ text, label = "출처" }: { text: string; label?: string }) {
  return (
    <details className="source">
      <summary>{label}</summary>
      <p className="src">{text}</p>
    </details>
  );
}

/**
 * A cause is judged only by its own self-test: unanswered stays "possible" (spec: don't assert
 * unless narrowed), the answer that implies it makes it "likely", any other answer "less".
 */
type CauseState = "possible" | "likely" | "less";

const STATE_LABEL: Record<CauseState, string> = {
  possible: "가능성",
  likely: "해당 가능성 높음",
  less: "자가 테스트 결과상 덜 해당",
};

export function causeState(cause: Cause, answer: string | undefined): CauseState {
  if (answer === undefined) return "possible";
  return answer === cause.self_test.implies_cause_on ? "likely" : "less";
}

function CauseCard({
  cause,
  answer,
  groupId,
  onAnswer,
}: {
  cause: Cause;
  answer: string | undefined;
  groupId: string;
  onAnswer: (optionId: string) => void;
}) {
  const [forceOpen, setForceOpen] = useState(false);
  const state = causeState(cause, answer);
  const open = state !== "less" || forceOpen;
  const st = cause.self_test;
  return (
    <div className={`cause${state === "less" ? " dim" : ""}`}>
      <h4>
        {cause.title}
        <span className="badge">{STATE_LABEL[state]}</span>
      </h4>
      <Source text={cause.source} label="원인 근거" />
      <fieldset className="selftest">
        <legend>30초 자가 테스트: {st.instruction}</legend>
        {st.options.map((o) => (
          <label className="radio" key={o.id}>
            <input
              type="radio"
              name={`st-${groupId}-${st.id}`}
              checked={answer === o.id}
              onChange={() => onAnswer(o.id)}
            />
            <span>{o.label}</span>
          </label>
        ))}
        <Source text={st.source} />
      </fieldset>
      {open ? (
        cause.exercises.map((ex) => (
          <div className="exercise" key={ex.id}>
            <strong>{ex.name}</strong>
            <p className="muted">{ex.sets_reps}</p>
            <p>{ex.description}</p>
            <Source text={ex.source} />
          </div>
        ))
      ) : (
        <button type="button" className="link" onClick={() => setForceOpen(true)}>
          운동 제안 펼치기
        </button>
      )}
    </div>
  );
}

export default function FindingsPanel({
  findings,
  metrics,
  analysisId,
}: {
  findings: Finding[];
  metrics: Metric[];
  analysisId: string;
}) {
  const [answers, setAnswers] = useState<Answers>(() => loadAnswers(analysisId));
  useEffect(() => setAnswers(loadAnswers(analysisId)), [analysisId]);

  const scored = metrics.filter((m) => m.score !== null).length;

  const answer = (selfTestId: string, optionId: string) =>
    setAnswers((prev) => {
      const next = { ...prev, [selfTestId]: optionId };
      saveAnswers(analysisId, next);
      return next;
    });

  return (
    <section className="card">
      <h2>진단과 개선 제안</h2>
      <p className="disclaimer">트레이닝 코치와 상의 후 적용하세요.</p>
      <p className="muted section-note">
        평가 가능한 지표 {scored}/{metrics.length}개
      </p>
      {findings.length === 0 &&
        (scored < MIN_SCORED_FOR_DIAGNOSIS ? (
          <p className="muted">평가된 지표가 부족해 진단하지 않았습니다.</p>
        ) : (
          <p className="muted">뚜렷한 문제 지점이 발견되지 않았습니다.</p>
        ))}
      {findings.map((f) => (
        <article className="finding" key={f.issue_id}>
          <h3>{f.title}</h3>
          <p>{f.evidence}</p>
          {f.causes.length === 0 ? (
            <p className="muted">근거가 확인된 원인 정보가 아직 없습니다.</p>
          ) : (
            f.causes.map((c) => (
              <CauseCard
                key={c.id}
                cause={c}
                answer={answers[c.self_test.id]}
                groupId={f.issue_id}
                onAnswer={(o) => answer(c.self_test.id, o)}
              />
            ))
          )}
        </article>
      ))}
    </section>
  );
}
