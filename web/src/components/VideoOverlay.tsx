import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AnalysisResult } from "../types";
import { analysisFps, frameAt, toCanvas } from "../render/frame";
import { activeLeaks, overlayConfidence, particlesFor, segmentStyle } from "../render/energy";
import { notifyOnChange, shortcutFor } from "../render/controls";
import { jointIndex, segmentJoints, segmentLines } from "../render/skeleton";
import type { Pt, Segment } from "../render/skeleton";

const SPEEDS = [0.1, 0.25, 0.5, 1];

/** A seek request; a new nonce re-seeks even to the same frame. */
export interface SeekTarget {
  frame: number;
  nonce: number;
}

interface Props {
  result: AnalysisResult;
  videoSrc: string;
  seekTo?: SeekTarget;
  onFrame?: (frame: number) => void;
}

export default function VideoOverlay({ result, videoSrc, seekTo, onFrame }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [speed, setSpeed] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [frame, setFrame] = useState(0);
  const [videoError, setVideoError] = useState(false);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;
  const notify = useMemo(() => notifyOnChange((f) => onFrameRef.current?.(f)), []);

  // fps: container rate, maps frames <-> playback time; motionFps: capture rate, for real-time spans
  const { fps, n_frames: nFrames, width, height } = result.meta;
  const motionFps = analysisFps(result.meta);
  const idx = useMemo(() => jointIndex(result.keypoints.joints), [result]);
  const joints = useMemo(() => segmentJoints(result.subject.hand), [result]);
  // dimming follows the confidence of the speed each column draws, not the segment score's
  const conf = useMemo(() => overlayConfidence(result.metrics), [result]);

  /** `mediaTime`: presentation time of the displayed frame (rVFC path); defaults to currentTime. */
  const draw = useCallback((mediaTime?: number) => {
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

    const t = mediaTime ?? video.currentTime;
    const f = frameAt(t, fps, nFrames);
    // fractional position inside the frame keeps particles moving during playback
    const fSmooth = Math.min(t * fps, nFrames - 1);
    const kp = result.keypoints.frames[f];
    if (!kp) return;
    const leaks = activeLeaks(result.energy.leaks, f, Math.round(0.05 * motionFps));
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
      const st = segmentStyle(intensity, conf[seg] ?? "low", isLeak);

      ctx.save();
      ctx.globalAlpha = st.alpha;
      ctx.strokeStyle = st.color;
      ctx.lineWidth = st.width;
      ctx.lineCap = st.cap;
      ctx.setLineDash(st.dash);
      if (st.glow) {
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

      // flowing particles (dimmed with the segment, and not additive, when confidence is low)
      ctx.save();
      ctx.globalCompositeOperation = st.additive ? "lighter" : "source-over";
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
    notify(f);
  }, [notify, result, idx, joints, conf, fps, motionFps, nFrames, width, height]);

  // redraw loop: requestVideoFrameCallback when available, else rAF
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let stop = false;
    let handle = 0;
    const rvfc = "requestVideoFrameCallback" in video;
    const loopRvfc = (_now: number, meta: { mediaTime: number }) => {
      if (stop) return;
      draw(meta.mediaTime);
      handle = video.requestVideoFrameCallback(loopRvfc);
    };
    const loopRaf = () => {
      if (stop) return;
      draw();
      handle = requestAnimationFrame(loopRaf);
    };
    if (rvfc) handle = video.requestVideoFrameCallback(loopRvfc);
    else loopRaf();
    draw();
    const redraw = () => draw();
    video.addEventListener("ratechange", redraw);
    video.addEventListener("seeked", redraw);
    video.addEventListener("loadeddata", redraw);
    window.addEventListener("resize", redraw);
    return () => {
      stop = true;
      if (rvfc) video.cancelVideoFrameCallback(handle);
      else cancelAnimationFrame(handle);
      video.removeEventListener("ratechange", redraw);
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

  const step = useCallback(
    (d: number) => {
      const v = videoRef.current;
      if (v) seek(frameAt(v.currentTime, fps, nFrames) + d);
    },
    [seek, fps, nFrames],
  );

  useEffect(() => {
    if (seekTo !== undefined) seek(seekTo.frame);
  }, [seekTo, seek]);

  useEffect(() => setVideoError(false), [videoSrc]);

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
    const sc = shortcutFor(e);
    if (!sc) return;
    e.preventDefault();
    if (sc === "prev") step(-1);
    else if (sc === "next") step(1);
    else toggle();
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
          onError={() => setVideoError(true)}
        />
        <canvas ref={canvasRef} />
      </div>
      {videoError && (
        <p className="error" role="alert">
          이 브라우저에서 영상을 재생할 수 없습니다 (예: HEVC .mov). MP4(H.264)로 변환해 올려 주세요.
        </p>
      )}
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
        <button type="button" className="ctl" onClick={() => step(-1)} aria-label="이전 프레임">◀</button>
        <button type="button" className="ctl" onClick={toggle}>{playing ? "정지" : "재생"}</button>
        <button type="button" className="ctl" onClick={() => step(1)} aria-label="다음 프레임">▶</button>
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
