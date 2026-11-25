import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getAnalysis, MAX_POLL_FAILURES, pollAnalysis, uploadVideo } from "./api";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("api", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("uploadVideo sends multipart fields", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ id: "abc123" }, 201));
    const file = new File(["x"], "pitch.mp4", { type: "video/mp4" });
    const id = await uploadVideo(file, { heightCm: 185, hand: "R", cameraSide: "glove" });
    expect(id).toBe("abc123");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/analyses");
    expect(init.method).toBe("POST");
    const fd = init.body as FormData;
    expect(fd.get("video")).toBeInstanceOf(File);
    expect(fd.get("height_cm")).toBe("185");
    expect(fd.get("hand")).toBe("R");
    expect(fd.get("camera_side")).toBe("glove");
  });

  it("uploadVideo sends capture_fps only when given", async () => {
    fetchMock.mockImplementation(async () => jsonResponse({ id: "abc123" }, 201));
    const file = new File(["x"], "pitch.mp4", { type: "video/mp4" });
    await uploadVideo(file, { heightCm: 185, hand: "R", cameraSide: "arm", captureFps: 240 });
    expect((fetchMock.mock.calls[0][1].body as FormData).get("capture_fps")).toBe("240");
    await uploadVideo(file, { heightCm: 185, hand: "R", cameraSide: "arm" });
    expect((fetchMock.mock.calls[1][1].body as FormData).has("capture_fps")).toBe(false);
  });

  it("uploadVideo throws the server detail on 422", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: "hand must be R or L" }, 422));
    const file = new File(["x"], "p.mp4");
    await expect(
      uploadVideo(file, { heightCm: 185, hand: "R", cameraSide: "arm" }),
    ).rejects.toThrow("hand must be R or L");
  });

  it("getAnalysis returns status, error, result", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ id: "a", status: "failed", error: "boom", result: null }),
    );
    expect(await getAnalysis("a")).toEqual({ status: "failed", error: "boom", result: null });
  });

  it("pollAnalysis stops on done", async () => {
    const seq = ["processing", "processing", "done"];
    let i = 0;
    fetchMock.mockImplementation(async () =>
      jsonResponse({ id: "a", status: seq[Math.min(i++, 2)], error: null, result: null }),
    );
    const updates: string[] = [];
    pollAnalysis("a", (s) => updates.push(s.status), 1000);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(updates).toEqual(["processing", "processing", "done"]);
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("pollAnalysis cancel stops polling", async () => {
    fetchMock.mockImplementation(async () =>
      jsonResponse({ id: "a", status: "processing", error: null, result: null }),
    );
    const onUpdate = vi.fn();
    const cancel = pollAnalysis("a", onUpdate, 1000);
    await vi.advanceTimersByTimeAsync(2500);
    const calls = fetchMock.mock.calls.length;
    expect(calls).toBeGreaterThanOrEqual(2);
    cancel();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(fetchMock.mock.calls.length).toBe(calls);
    const updatesAtCancel = onUpdate.mock.calls.length;
    await vi.advanceTimersByTimeAsync(5000);
    expect(onUpdate.mock.calls.length).toBe(updatesAtCancel);
  });

  it("pollAnalysis reports 404 once and stops", async () => {
    fetchMock.mockImplementation(async () => jsonResponse({ detail: "analysis not found" }, 404));
    const onError = vi.fn();
    pollAnalysis("zzz", vi.fn(), 1000, onError);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0].status).toBe(404);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("pollAnalysis gives up after persistent failures but retries transient ones", async () => {
    let n = 0;
    fetchMock.mockImplementation(async () => {
      n += 1;
      if (n === 1) throw new TypeError("Failed to fetch");
      return jsonResponse({ id: "a", status: "done", error: null, result: null });
    });
    const onError = vi.fn();
    const updates: string[] = [];
    pollAnalysis("a", (s) => updates.push(s.status), 1000, onError);
    await vi.advanceTimersByTimeAsync(5000);
    expect(updates).toEqual(["done"]);
    expect(onError).not.toHaveBeenCalled();

    fetchMock.mockReset();
    fetchMock.mockImplementation(async () => {
      throw new TypeError("Failed to fetch");
    });
    pollAnalysis("b", vi.fn(), 1000, onError);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(fetchMock).toHaveBeenCalledTimes(MAX_POLL_FAILURES);
    expect(onError).toHaveBeenCalledTimes(1);
  });
});
