export interface Size {
  w: number;
  h: number;
}

/**
 * Frames per second of the captured motion: use it to turn frame counts into milliseconds.
 * `fps` (the container's) maps frames to playback time instead; the two differ for slow-motion
 * clips exported at 30 fps.
 */
export function analysisFps(meta: { fps: number; capture_fps: number | null }): number {
  return meta.capture_fps ?? meta.fps;
}

/** Video time → frame index, clamped to the valid range. */
export function frameAt(timeS: number, fps: number, nFrames: number): number {
  const f = Math.round(timeS * fps);
  return Math.min(Math.max(f, 0), Math.max(nFrames - 1, 0));
}

/** Video pixel → canvas pixel, matching `object-fit: contain` letterboxing. */
export function toCanvas(x: number, y: number, video: Size, canvas: Size): [number, number] {
  const scale = Math.min(canvas.w / video.w, canvas.h / video.h);
  const offX = (canvas.w - video.w * scale) / 2;
  const offY = (canvas.h - video.h * scale) / 2;
  return [offX + x * scale, offY + y * scale];
}
