import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const upload = vi.fn();
vi.mock("../api", async (orig) => ({
  ...(await orig<typeof import("../api")>()),
  listAnalyses: async () => [],
  uploadVideo: (...a: unknown[]) => upload(...a),
}));

import UploadPage from "./UploadPage";

async function fill(captureFps: string) {
  const user = userEvent.setup();
  render(<UploadPage />);
  const file = new File(["x"], "pitch.mp4", { type: "video/mp4" });
  await user.upload(screen.getByLabelText("영상 파일"), file);
  await user.type(screen.getByLabelText("키 (cm)"), "176");
  if (captureFps) await user.type(screen.getByLabelText(/촬영 fps/), captureFps);
  await user.click(screen.getByRole("button", { name: "분석 시작" }));
}

describe("UploadPage capture fps", () => {
  beforeEach(() => {
    upload.mockReset();
    upload.mockResolvedValue("abc");
  });

  it("is optional", async () => {
    await fill("");
    expect(upload).toHaveBeenCalledTimes(1);
    expect(upload.mock.calls[0][1].captureFps).toBeUndefined();
  });

  it("is sent when given", async () => {
    await fill("240");
    expect(upload.mock.calls[0][1]).toMatchObject({ heightCm: 176, captureFps: 240 });
  });

  it("must be between 1 and 1000", async () => {
    await fill("2000");
    expect(upload).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("촬영 fps는 1~1000 사이");
  });
});
