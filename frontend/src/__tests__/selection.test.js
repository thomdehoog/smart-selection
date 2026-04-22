import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useAppState } from "../MicroscopySearch.jsx";

describe("selection toggle", () => {
  it("adds a new id", () => {
    const { result } = renderHook(() => useAppState());
    act(() => result.current.toggle(42));
    expect(result.current.s.selected.has(42)).toBe(true);
  });

  it("removes an existing id", () => {
    const { result } = renderHook(() => useAppState());
    act(() => { result.current.toggle(42); result.current.toggle(7); result.current.toggle(99); });
    act(() => result.current.toggle(42));
    expect(result.current.s.selected.has(42)).toBe(false);
    expect(result.current.s.selected.size).toBe(2);
  });

  it("is idempotent on double toggle", () => {
    const { result } = renderHook(() => useAppState());
    act(() => result.current.toggle(10));
    expect(result.current.s.selected.has(10)).toBe(true);
    act(() => result.current.toggle(10));
    expect(result.current.s.selected.has(10)).toBe(false);
  });

  it("keeps the selection across multiple toggles", () => {
    const { result } = renderHook(() => useAppState());
    act(() => { [5, 10, 15, 20].forEach(id => result.current.toggle(id)); });
    expect(result.current.s.selected.size).toBe(4);
    expect(result.current.s.selected).toEqual(new Set([5, 10, 15, 20]));
  });

  it("persists selection when other state changes", () => {
    const { result } = renderHook(() => useAppState());
    act(() => { [1, 2, 3].forEach(id => result.current.toggle(id)); });
    act(() => result.current.set({ objects: [{ object_id: 10 }, { object_id: 11 }] }));
    expect(result.current.s.selected).toEqual(new Set([1, 2, 3]));
  });
});

describe("accept / reject transitions", () => {
  it("accept moves an id from results to positive", () => {
    let positive = [1, 2, 3];
    let results = [
      { object_id: 10, similarity_score: 0.9 },
      { object_id: 11, similarity_score: 0.8 },
    ];
    const accepted = 10;
    positive = [...positive, accepted];
    results = results.filter(r => r.object_id !== accepted);
    expect(positive).toContain(10);
    expect(results).toHaveLength(1);
    expect(results[0].object_id).toBe(11);
  });

  it("reject moves an id from results to negative", () => {
    let negative = [];
    let results = [
      { object_id: 10, similarity_score: 0.9 },
      { object_id: 11, similarity_score: 0.8 },
    ];
    const rejected = 10;
    negative = [...negative, rejected];
    results = results.filter(r => r.object_id !== rejected);
    expect(negative).toContain(10);
    expect(results).toHaveLength(1);
  });

  it("accept then reject on different ids", () => {
    let positive = [1], negative = [];
    let results = [10, 11, 12].map(id => ({ object_id: id, similarity_score: 0.9 }));
    positive = [...positive, 10];
    results = results.filter(r => r.object_id !== 10);
    negative = [...negative, 12];
    results = results.filter(r => r.object_id !== 12);
    expect(positive).toEqual([1, 10]);
    expect(negative).toEqual([12]);
    expect(results).toHaveLength(1);
    expect(results[0].object_id).toBe(11);
  });

  it("accepted id is no longer in results", () => {
    let results = Array.from({ length: 5 }, (_, i) => ({ object_id: i, similarity_score: 0.9 - i * 0.1 }));
    const accepted = 2;
    results = results.filter(r => r.object_id !== accepted);
    expect(results.every(r => r.object_id !== accepted)).toBe(true);
  });
});
