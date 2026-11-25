import { useCallback, useEffect, useState } from "react";
import { HttpError, pollAnalysis, videoUrl } from "../api";
import type { AnalysisResult, AnalysisState, Metric, VideoMeta } from "../types";
import { CONFIDENCE_LABEL, missingReason } from "../render/timeline";
import FilmingGuide from "./FilmingGuide";
import FindingsPanel from "./FindingsPanel";
import PhaseTimeline from "./PhaseTimeline";
import SegmentScores from "./SegmentScores";
import VideoOverlay from "./VideoOverlay";
import type { SeekTarget } from "./VideoOverlay";

function fmt(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

function fpsText(v: number | null): string {
  if (v === null || !Number.isFinite(v) || v <= 0) return "fps 알 수 없음";
  return `${Number.isInteger(v) ? v : Number(v.toFixed(2))}fps`;
}

/** Container fps (playback), analysis fps (capture) and the fps of the norms used. */
function FpsLine({ meta }: { meta: VideoMeta }) {
  const parts = [`영상 ${fpsText(meta.fps)}`, `분석 ${fpsText(meta.capture_fps)}`];
  if (meta.norm_fps !== null) parts.push(`비교 기준값 ${fpsText(meta.norm_fps)}`);
  return <p className="muted section-note">{parts.join(" · ")}</p>;
}

function Warnings({ result }: { result: AnalysisResult }) {
  if (result.warnings.length === 0) return null;
  const sev = { error: "⛔ 오류", warn: "⚠ 주의", info: "ℹ 안내" } as const;
  const order = { error: 0, warn: 1, info: 2 } as const;
  const sorted = [...result.warnings].sort((a, b) => order[a.severity] - order[b.severity]);
  return (
    <section className="card" aria-label="경고">
      <h2>확인할 점</h2>
      <ul className="warn-list">
        {sorted.map((w, i) => (
          <li key={`${w.code}-${i}`} className={`warn-item ${w.severity}`}>
            <strong>{sev[w.severity]}</strong> {w.message}
          </li>
        ))}
      </ul>
    </section>
  );
}

function MetricsTable({ result }: { result: AnalysisResult }) {
  return (
    <section className="card">
      <h2>지표</h2>
      <div className="table-wrap">
        <table className="metrics">
          <thead>
            <tr>
              <th>지표</th>
              <th>값</th>
              <th>점수</th>
              <th>신뢰도</th>
            </tr>
          </thead>
          <tbody>
            {result.metrics.map((m: Metric) => (
              <tr key={m.name}>
                <td>
                  {m.label}
                  {m.norm ? (
                    <details>
                      <summary>출처</summary>
                      <p className="src">{m.norm.source}</p>
                    </details>
                  ) : (
                    <span className="why">비교 기준 없음</span>
                  )}
                </td>
                <td className="num" data-label="값">
                  {m.value === null ? (
                    <>
                      —<span className="why">{missingReason(m, result.warnings)}</span>
                    </>
                  ) : (
                    `${fmt(m.value)}${m.unit ? ` ${m.unit}` : ""}`
                  )}
                </td>
                <td className="num" data-label="점수">{m.score === null ? "참고값" : Math.round(m.score)}</td>
                <td data-label="신뢰도">
                  <span className={`badge conf-${m.confidence}`}>{CONFIDENCE_LABEL[m.confidence]}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function AnalysisPage({ id }: { id: string }) {
  const [state, setState] = useState<AnalysisState | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  const [seekTo, setSeekTo] = useState<SeekTarget | undefined>(undefined);

  useEffect(() => {
    setState(null);
    setLoadError(null);
    setFrame(0);
    setSeekTo(undefined);
    return pollAnalysis(id, setState, 1000, (err) => {
      setLoadError(
        err instanceof HttpError && err.status === 404
          ? "분석을 찾을 수 없습니다."
          : "서버에 연결할 수 없습니다. 분석 서버가 실행 중인지 확인해 주세요.",
      );
    });
  }, [id]);

  const onFrame = useCallback((f: number) => setFrame(f), []);
  // a fresh object per request, so choosing the same phase again seeks again
  const onSeek = useCallback(
    (f: number) => setSeekTo((prev) => ({ frame: f, nonce: (prev?.nonce ?? 0) + 1 })),
    [],
  );

  const result = state?.status === "done" ? state.result : null;
  const hasError = result?.warnings.some((w) => w.severity === "error") ?? false;
  const hasWarn = result?.warnings.some((w) => w.severity === "warn") ?? false;

  let body;
  if (loadError) {
    body = <p className="error" role="alert">{loadError}</p>;
  } else if (state === null) {
    body = <p className="muted" role="status">불러오는 중…</p>;
  } else if (state.status === "processing") {
    body = (
      <>
        <p className="muted" role="status">분석 중입니다. 영상 길이와 fps에 따라 수십 초~수 분 걸립니다…</p>
        <div className="progress" role="progressbar" aria-label="분석 진행 중" />
      </>
    );
  } else if (state.status === "failed") {
    body = (
      <p className="error" role="alert">
        분석에 실패했습니다{state.error ? `: ${state.error}` : "."}
      </p>
    );
  } else if (!result) {
    body = <p className="error" role="alert">결과를 불러오지 못했습니다.</p>;
  } else if (hasError) {
    body = (
      <>
        <Warnings result={result} />
        <FilmingGuide />
        <p className="disclaimer">트레이닝 코치와 상의 후 적용하세요.</p>
      </>
    );
  } else {
    body = (
      <>
        {hasWarn && (
          <p className="muted section-note">아래 경고 때문에 일부 값은 비어 있거나 참고용입니다.</p>
        )}
        <Warnings result={result} />
        <section className="card">
          <VideoOverlay result={result} videoSrc={videoUrl(id)} seekTo={seekTo} onFrame={onFrame} />
        </section>
        <PhaseTimeline result={result} currentFrame={frame} onSeek={onSeek} />
        <SegmentScores segments={result.segments} />
        <MetricsTable result={result} />
        <FindingsPanel findings={result.findings} metrics={result.metrics} analysisId={id} />
      </>
    );
  }

  return (
    <>
      <a href="#/" className="back">← 처음으로</a>
      <h1>분석 결과</h1>
      {result && <FpsLine meta={result.meta} />}
      {body}
    </>
  );
}
