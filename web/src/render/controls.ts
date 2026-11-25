export type Shortcut = "prev" | "next" | "toggle";

/** Keyboard shortcut for an event; only when the wrapper itself is the target (not inner buttons/inputs). */
export function shortcutFor(e: { key: string; target: unknown; currentTarget: unknown }): Shortcut | null {
  if (e.target !== e.currentTarget) return null;
  if (e.key === "ArrowLeft") return "prev";
  if (e.key === "ArrowRight") return "next";
  if (e.key === " ") return "toggle";
  return null;
}

/** Returns a function that forwards a frame index to `cb` only when it differs from the last one. */
export function notifyOnChange(cb: (frame: number) => void): (frame: number) => void {
  let last: number | null = null;
  return (frame) => {
    if (frame === last) return;
    last = frame;
    cb(frame);
  };
}
