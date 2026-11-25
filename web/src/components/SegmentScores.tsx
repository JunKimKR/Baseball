import type { SegmentScore } from "../types";
import { CONFIDENCE_LABEL } from "../render/timeline";

export default function SegmentScores({ segments }: { segments: SegmentScore[] }) {
  return (
    <section className="card">
      <h2>부위별 점수</h2>
      {segments.map((s) => (
        <div className="seg-row" key={s.segment}>
          <span>{s.label}</span>
          <div className="seg-track" role="presentation">
            <div
              className={`seg-fill${s.score === null ? " ref" : ""}`}
              style={{ width: `${s.score === null ? 100 : Math.max(0, Math.min(100, s.score))}%` }}
            />
          </div>
          <span className="seg-val">
            {s.score === null ? "참고값" : Math.round(s.score)}{" "}
            <span className={`badge conf-${s.confidence}`}>신뢰도 {CONFIDENCE_LABEL[s.confidence]}</span>
          </span>
        </div>
      ))}
      <p className="muted section-note">점수는 참고 데이터 대비 상대 위치이며 진단이 아닙니다.</p>
    </section>
  );
}
