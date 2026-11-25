import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import fixture from "../fixtures/sample-result.json";
import type { AnalysisResult } from "../types";
import VideoOverlay from "./VideoOverlay";

const r = fixture as unknown as AnalysisResult;

describe("VideoOverlay", () => {
  beforeEach(() => {
    // jsdom has no 2D canvas; the overlay skips drawing without a context
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  });

  it("explains a video the browser cannot play", () => {
    const { container } = render(<VideoOverlay result={r} videoSrc="/v.mov" />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    fireEvent.error(container.querySelector("video")!);
    expect(screen.getByRole("alert")).toHaveTextContent(
      "이 브라우저에서 영상을 재생할 수 없습니다 (예: HEVC .mov). MP4(H.264)로 변환해 올려 주세요.",
    );
  });
});
