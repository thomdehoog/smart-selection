import { describe, it, expect, vi, afterEach } from "vitest";
import { api } from "../MicroscopySearch.jsx";

afterEach(() => vi.unstubAllGlobals());

function mockFetch(response) {
  vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(response)));
}

describe("apiFetch error handling", () => {
  it("returns the parsed body on 2xx", async () => {
    mockFetch({
      ok: true, status: 200, statusText: "OK",
      json: async () => ({ dataset_id: "abc", num_images: 3 }),
    });
    const r = await api.upload("/some/path", 3);
    expect(r.dataset_id).toBe("abc");
    expect(r.num_images).toBe(3);
  });

  it("throws the server's error message on non-2xx with JSON body", async () => {
    mockFetch({
      ok: false, status: 400, statusText: "Bad Request",
      json: async () => ({ error: "Invalid image directory: /nope" }),
    });
    await expect(api.upload("/nope", 1)).rejects.toThrow(
      "Invalid image directory: /nope",
    );
  });

  it("falls back to status text when the body has no error field", async () => {
    mockFetch({
      ok: false, status: 500, statusText: "Internal Server Error",
      json: async () => ({}),
    });
    await expect(api.status()).rejects.toThrow("500 Internal Server Error");
  });

  it("falls back to status when the body is non-JSON", async () => {
    mockFetch({
      ok: false, status: 503, statusText: "Service Unavailable",
      json: async () => { throw new Error("not json"); },
    });
    await expect(api.status()).rejects.toThrow("503 Service Unavailable");
  });
});
