import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useAppState } from "../MicroscopySearch.jsx";

describe("step navigation", () => {
  it("initial state is step 1", () => {
    const { result } = renderHook(() => useAppState());
    expect(result.current.s.step).toBe(1);
    expect(result.current.s.datasetId).toBeNull();
    expect(result.current.s.selected.size).toBe(0);
  });

  it("load → segmentation", () => {
    const { result } = renderHook(() => useAppState());
    act(() => result.current.set({ datasetId: "abc", step: 2 }));
    expect(result.current.s.step).toBe(2);
  });

  it("step 3 similarity requires a selection before search", () => {
    const { result } = renderHook(() => useAppState());
    expect(result.current.s.selected.size).toBe(0);
    act(() => result.current.toggle(1));
    expect(result.current.s.selected.size).toBe(1);
  });

  it("pipeline status gates step 3 UI", () => {
    const { result } = renderHook(() => useAppState());
    expect(result.current.s.pipelineStatus).toBe("idle");
    act(() => result.current.set({ pipelineStatus: "running", progress: 0.4 }));
    expect(result.current.s.pipelineStatus).toBe("running");
    expect(result.current.s.progress).toBe(0.4);
    act(() => result.current.set({ pipelineStatus: "done" }));
    expect(result.current.s.pipelineStatus).toBe("done");
  });

  it("resetting search state preserves selection", () => {
    const { result } = renderHook(() => useAppState());
    act(() => {
      result.current.toggle(1);
      result.current.toggle(2);
      result.current.set({ positive: [1, 2], negative: [5, 6], results: [{ object_id: 10 }] });
    });
    act(() => result.current.set({ negative: [], results: [] }));
    expect(result.current.s.negative).toEqual([]);
    expect(result.current.s.results).toEqual([]);
    expect(result.current.s.positive).toEqual([1, 2]);
    expect(result.current.s.selected).toEqual(new Set([1, 2]));
  });
});
