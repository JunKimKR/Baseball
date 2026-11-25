import { useMemo } from "react";
import type { AnalysisResult } from "../types";
import { analysisFps } from "../render/frame";
import { PHASE_STATUS_LABEL, ghostMarkers, phaseRects, phaseStatus } from "../render/timeline";
import type { PhaseStatus } from "../render/timeline";

interface Props {
  result: AnalysisResult;
  currentFrame: number;
  onSeek: (frame: number) => void;
}

const EVENT_LABEL: Array<[keyof AnalysisResult["events"], string]> = [
  ["leg_lift_peak", "레그 리프트 정점"],
  ["foot_plant", "발 착지"],
  ["mer_est", "최대 외회전(추정)"],
  ["release", "릴리스"],
];

const STATUS_ICON: Record<PhaseStatus, string> = { ok: "✓", warn: "▲", unevaluated: "–" };
const STATUS_CLASS: Record<PhaseStatus, string> = {
  ok: "status-ok",
  warn: "status-warn",
  unevaluated: "status-none",
};

function msDiff(actual: number, ideal: number, fps: number): string {
  const ms = Math.round(((actual - ideal) / fps) * 1000);
  return `${ms > 0 ? "+" : ""}${ms}ms`;
}

export default function PhaseTimeline({ result, currentFrame, onSeek }: Props) {
  const { meta, phases, metrics, energy, events } = result;
  const span = Math.max(meta.n_frames - 1, 1);
  const pct = (f: number) => `${(f / span) * 100}%`;
  const rects = useMemo(() => phaseRects(phases, meta.n_frames, 100), [phases, meta.n_frames]);
  const fps = analysisFps(meta); // frames <-> ms in capture time (slow-motion exports)
  const ghosts = useMemo(() => ghostMarkers(metrics, energy.peaks, fps), [metrics, energy.peaks, fps]);
  const status = (name: string) => {
    const p = phases.find((x) => x.name === name)!;
    return phaseStatus(p, metrics, energy.leaks);
  };

  return (
    <section className="card">
      <h2>동작 구간</h2>
      <p className="muted section-note">구간을 누르면 해당 프레임으로 이동합니다. 아래 목록으로도 이동할 수 있습니다.</p>
      <div className="timeline">
        <div className="tl-bar">
          {rects.map((r) => {
            const p = phases.find((x) => x.name === r.name)!;
            const st = status(r.name);
            return (
              <button
                type="button"
                key={r.name}
                className={`tl-phase ${st}`}
                style={{ left: `${r.x}%`, width: `${r.w}%` }}
                title={`${r.label} (${PHASE_STATUS_LABEL[st]})`}
                aria-label={`${r.label} 구간으로 이동 — ${PHASE_STATUS_LABEL[st]}`}
                onClick={() => onSeek(p.start)}
              >
                {r.label}
              </button>
            );
          })}
          {EVENT_LABEL.map(([key, label]) => {
            const f = events[key];
            return f === null ? null : (
              <span key={key}>
                <span className="tl-event" style={{ left: pct(f) }} title={`${label} (프레임 ${f})`} />
              </span>
            );
          })}
          {energy.leaks.map((l, i) => (
            <span
              key={i}
              className="tl-leak"
              style={{ left: pct(l.frame) }}
              title={`에너지 누수: ${l.message}`}
              role="img"
              aria-label={`에너지 누수 (프레임 ${l.frame})`}
            />
          ))}
          <span className="tl-cursor" style={{ left: pct(currentFrame) }} aria-hidden="true" />
        </div>

        {ghosts.length > 0 && (
          <div className="tl-ghost-row" aria-hidden="true">
            {ghosts.map((g) => (
              <span key={g.label}>
                <span className="tl-dot ghost" style={{ left: pct(g.idealFrame) }} />
                <span className="tl-dot actual" style={{ left: pct(g.actualFrame) }} />
              </span>
            ))}
          </div>
        )}
      </div>
      <ul className="phase-list">
        {phases.map((p) => {
          const st = status(p.name);
          return (
            <li key={p.name}>
              <button type="button" className="phase-btn" onClick={() => onSeek(p.start)}>
                <span className={STATUS_CLASS[st]}>{STATUS_ICON[st]} {PHASE_STATUS_LABEL[st]}</span>{" "}
                {p.label} <span className="muted">프레임 {p.start}–{p.end}</span>
              </button>
            </li>
          );
        })}
      </ul>
      {ghosts.length > 0 && (
        <ul className="phase-list" aria-label="타이밍 비교">
          {ghosts.map((g) => (
            <li key={g.label}>
              {g.label} 최고속도 시점: 기준보다 <strong>{msDiff(g.actualFrame, g.idealFrame, fps)}</strong>
            </li>
          ))}
        </ul>
      )}
      <ul className="tl-legend">
        <li>
          <span className="status-warn">■</span> 노란 밑줄: 점검이 필요한 구간
        </li>
        <li>
          <span className="status-none">–</span> 평가 안 됨: 점수를 낼 수 있는 지표가 없는 구간(양호하다는 뜻이 아님)
        </li>
        <li>
          <span style={{ color: "var(--energy-high)" }}>●</span> 이벤트:{" "}
          {EVENT_LABEL.filter(([k]) => events[k] !== null).map(([, l]) => l).join(" · ")}
        </li>
        <li>
          <span style={{ color: "var(--leak)" }}>●</span> 빨강: 에너지 누수 지점
        </li>
        {ghosts.length > 0 && <li>아래 줄: ● 실제 · ○ 기준(반투명) 최고속도 시점 — 골반 최고 시점부터 기준 간격을 누적</li>}
      </ul>
    </section>
  );
}
