import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AnalysisResult, Confidence } from "../types";
import { frameAt, toCanvas } from "../render/frame";
import { activeLeaks, particlesFor, segmentStyle } from "../render/energy";
import { jointIndex, segmentJoints, segmentLines } from "../render/skeleton";
import type { Pt, Segment } from "../render/skeleton";

const SPEEDS = [0.1, 0.25, 0.5, 1];

interface Props {
  result: AnalysisResult;
  videoSrc: string;
  seekToFrame?: number;
  onFrame?: (frame: number) => void;
}

export default function VideoOverlay({ result, videoSrc, seekToFrame, onFrame }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [speed, setSpeed] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [frame, setFrame] = useState(0);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  const { fps, n_frames: nFrames, width, height } = result.meta;
  const idx = useMemo(() => jointIndex(result.keypoints.joints), [result]);
  const joints = useMemo(() => segmentJoints(result.subject.hand), [result]);
  const conf = useMemo(() => {
    const m: Record<string, Confidence> = {};
    for (const s of result.segments) m[s.segment] = s.confidence;
    return m;
  }, [result]);

  const draw = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(ch * dpr);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    const f = frameAt(video.currentTime, fps, nFrames);
    // fractional position inside the frame keeps particles moving during playback
    const fSmooth = Math.min(video.currentTime * fps, nFrames - 1);
    const kp = result.keypoints.frames[f];
    if (!kp) return;
    const leaks = activeLeaks(result.energy.leaks, f, Math.round(0.05 * fps));
    const leaking = new Set(leaks.map((l) => l.segment));
    const vs = { w: width, h: height };
    const cs = { w: cw, h: ch };
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    result.energy.segments.forEach((name, si) => {
      const seg = name as Segment;
      const pairs = joints[seg];
      if (!pairs) return;
      const lines = segmentLines(kp, idx, pairs).map(
        ([a, b]) => [toCanvas(a[0], a[1], vs, cs), toCanvas(b[0], b[1], vs, cs)] as [Pt, Pt],
      );
      if (lines.length === 0) return;
      const intensity = result.energy.intensity[f]?.[si] ?? 0;
      const isLeak = leaking.has(name);
      const st = segmentStyle(intensity, conf[name] ?? "low", isLeak);

      ctx.save();
      ctx.globalAlpha = st.alpha;
      ctx.strokeStyle = st.color;
      ctx.lineWidth = st.width;
      ctx.setLineDash(st.dash);
      if (intensity > 0.5 || isLeak) {
        ctx.globalCompositeOperation = "lighter";
        ctx.shadowColor = st.color;
        ctx.shadowBlur = 6 + 22 * intensity;
      }
      ctx.beginPath();
      for (const [a, b] of lines) {
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }
      ctx.stroke();
      ctx.restore();

      // flowing particles (dimmed with the segment when confidence is low)
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      ctx.fillStyle = isLeak ? st.color : "#fff7c2";
      for (const p of particlesFor(name, intensity, fSmooth, 1, { line: lines, leaking: isLeak })) {
        ctx.globalAlpha = p.alpha * st.alpha;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    });

    setFrame((prev) => (prev === f ? prev : f));
    onFrameRef.current?.(f);
  }, [result, idx, joints, conf, fps, nFrames, width, height]);

  // redraw loop: requestVideoFrameCallback when available, else rAF
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let stop = false;
    let handle = 0;
    const rvfc = "requestVideoFrameCallback" in video;
    const loop = () => {
      if (stop) return;
      draw();
      handle = rvfc
        ? (video as HTMLVideoElement).requestVideoFrameCallback(loop)
        : requestAnimationFrame(loop);
    };
    loop();
    const redraw = () => draw();
    video.addEventListener("seeked", redraw);
    video.addEventListener("loadeddata", redraw);
    window.addEventListener("resize", redraw);
    return () => {
      stop = true;
      if (rvfc) video.cancelVideoFrameCallback(handle);
      else cancelAnimationFrame(handle);
      video.removeEventListener("seeked", redraw);
      video.removeEventListener("loadeddata", redraw);
      window.removeEventListener("resize", redraw);
    };
  }, [draw]);

  const seek = useCallback(
    (f: number) => {
      const v = videoRef.current;
      if (!v) return;
      const c = Math.min(Math.max(f, 0), nFrames - 1);
      v.pause();
      // aim a quarter frame past the boundary so frameAt() rounds back to c
      v.currentTime = (c + 0.25) / fps;
    },
    [fps, nFrames],
  );

  useEffect(() => {
    if (seekToFrame !== undefined) seek(seekToFrame);
  }, [seekToFrame, seek]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = speed;
  }, [speed]);

  const toggle = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      seek(frame - 1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      seek(frame + 1);
    } else if (e.key === " ") {
      e.preventDefault();
      toggle();
    }
  };

  return (
    <div className="overlay" tabIndex={0} onKeyDown={onKeyDown} aria-label="영상 오버레이">
      <div className="overlay-stage" ref={wrapRef} style={{ aspectRatio: `${width} / ${height}` }}>
        <video
          ref={videoRef}
          src={videoSrc}
          muted
          playsInline
          preload="auto"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
        />
        <canvas ref={canvasRef} />
      </div>
      <input
        className="overlay-scrub"
        type="range"
        min={0}
        max={nFrames - 1}
        value={frame}
        onChange={(e) => seek(Number(e.target.value))}
        aria-label="프레임 이동"
      />
      <div className="overlay-controls">
        <button type="button" className="ctl" onClick={() => seek(frame - 1)} aria-label="이전 프레임">◀</button>
        <button type="button" className="ctl" onClick={toggle}>{playing ? "정지" : "재생"}</button>
        <button type="button" className="ctl" onClick={() => seek(frame + 1)} aria-label="다음 프레임">▶</button>
        <span className="muted overlay-frame">프레임 {frame} / {nFrames - 1}</span>
      </div>
      <div className="overlay-speeds" role="group" aria-label="재생 속도">
        {SPEEDS.map((s) => (
          <button
            type="button"
            key={s}
            className={"ctl" + (s === speed ? " active" : "")}
            aria-pressed={s === speed}
            onClick={() => setSpeed(s)}
          >
            {s}×
          </button>
        ))}
      </div>
      <p className="muted overlay-hint">←/→ 한 프레임, 스페이스 재생/정지</p>
    </div>
  );
}
