import { useEffect, useState, type FormEvent } from "react";
import { listAnalyses, uploadVideo } from "../api";
import type { AnalysisSummary, CameraSide, Hand } from "../types";

const STATUS_LABEL: Record<string, string> = {
  processing: "분석 중",
  done: "완료",
  failed: "실패",
};

const DETAIL_KO: Array<[RegExp, string]> = [
  [/hand must be/i, "투구 손은 오른손 또는 왼손이어야 합니다."],
  [/camera_side must be/i, "카메라 방향 값이 올바르지 않습니다."],
  [/height_cm must be/i, "키는 100~250cm 사이로 입력해 주세요."],
  [/unsupported video type/i, "지원하지 않는 영상 형식입니다. mp4, mov 등 동영상 파일을 올려 주세요."],
  [/empty video/i, "비어 있는 영상 파일입니다. 다른 파일을 선택해 주세요."],
];

export function koreanError(message: string): string {
  for (const [re, ko] of DETAIL_KO) if (re.test(message)) return ko;
  if (/failed to fetch|networkerror|load failed/i.test(message)) {
    return "서버에 연결할 수 없습니다. 분석 서버가 실행 중인지 확인해 주세요.";
  }
  return message;
}

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [height, setHeight] = useState("");
  const [hand, setHand] = useState<Hand>("R");
  const [side, setSide] = useState<CameraSide>("arm");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<AnalysisSummary[]>([]);

  useEffect(() => {
    let alive = true;
    listAnalyses()
      .then((l) => alive && setHistory(l))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const h = Number(height);
    if (!file) return setError("영상 파일을 선택해 주세요.");
    if (!Number.isFinite(h) || h < 100 || h > 250) {
      return setError("키는 100~250cm 사이로 입력해 주세요.");
    }
    setBusy(true);
    try {
      const id = await uploadVideo(file, { heightCm: h, hand, cameraSide: side });
      window.location.hash = `#/a/${id}`;
    } catch (err) {
      setError(koreanError(err instanceof Error ? err.message : String(err)));
      setBusy(false);
    }
  }

  return (
    <>
      <header>
        <h1>PitchCoach</h1>
        <p className="muted">투구 영상을 올리면 에너지가 어디서 새는지 보여드려요.</p>
      </header>

      <section className="card">
        <h2>촬영 가이드</h2>
        <ul className="guide">
          <li>투수의 옆모습이 보이도록 투구 방향과 90° 각도로 촬영하세요.</li>
          <li>삼각대로 카메라를 고정하세요. 손으로 들고 찍으면 속도 값이 부정확해집니다.</li>
          <li>머리부터 발끝까지 전신이 프레임 안에 들어와야 합니다.</li>
          <li>슬로모션(120~240fps)으로 촬영하면 더 정확합니다.</li>
        </ul>
      </section>

      <form className="card" onSubmit={onSubmit}>
        <h2>새 분석</h2>
        <label className="field">
          <span>영상 파일</span>
          <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </label>
        <label className="field">
          <span>키 (cm)</span>
          <input
            type="number"
            inputMode="decimal"
            min={100}
            max={250}
            step="0.1"
            placeholder="예: 180"
            value={height}
            onChange={(e) => setHeight(e.target.value)}
          />
        </label>
        <fieldset className="field">
          <legend>투구 손</legend>
          <label className="radio">
            <input type="radio" name="hand" checked={hand === "R"} onChange={() => setHand("R")} /> 오른손
          </label>
          <label className="radio">
            <input type="radio" name="hand" checked={hand === "L"} onChange={() => setHand("L")} /> 왼손
          </label>
        </fieldset>
        <fieldset className="field">
          <legend>카메라 방향</legend>
          <label className="radio">
            <input type="radio" name="side" checked={side === "arm"} onChange={() => setSide("arm")} /> 투구 팔 쪽
          </label>
          <label className="radio">
            <input type="radio" name="side" checked={side === "glove"} onChange={() => setSide("glove")} /> 글러브 쪽
          </label>
        </fieldset>
        {error && <p className="error" role="alert">{error}</p>}
        <button type="submit" disabled={busy}>{busy ? "올리는 중…" : "분석 시작"}</button>
      </form>

      <section className="card">
        <h2>기록</h2>
        {history.length === 0 ? (
          <p className="muted">아직 분석 기록이 없습니다.</p>
        ) : (
          <ul className="history">
            {history.map((a) => (
              <li key={a.id}>
                <a href={`#/a/${a.id}`}>{formatDate(a.created_at)}</a>
                <span className={`badge ${a.status}`}>{STATUS_LABEL[a.status] ?? a.status}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString("ko-KR");
}
