import { describe, it, expect } from "vitest";
import { scoreColor } from "../MicroscopySearch.jsx";

describe("threshold filtering", () => {
  const all = [
    { object_id: 1, similarity_score: 0.95 },
    { object_id: 2, similarity_score: 0.5 },
    { object_id: 3, similarity_score: 0.1 },
  ];

  it("threshold 0 shows all", () => {
    expect(all.filter(r => r.similarity_score >= 0)).toHaveLength(3);
  });

  it("threshold 0.6 filters low scores", () => {
    const visible = all.filter(r => r.similarity_score >= 0.6);
    expect(visible).toHaveLength(1);
    expect(visible[0].object_id).toBe(1);
  });

  it("threshold 1 shows none", () => {
    expect(all.filter(r => r.similarity_score >= 1)).toHaveLength(0);
  });

  it("threshold boundary is inclusive", () => {
    const r = [{ object_id: 1, similarity_score: 0.5 }];
    expect(r.filter(x => x.similarity_score >= 0.5)).toHaveLength(1);
  });
});

describe("scoreColor", () => {
  it("high scores are green", () => {
    expect(scoreColor(0.95)).toBe("#16A34A");
    expect(scoreColor(0.81)).toBe("#16A34A");
  });

  it("medium scores are amber", () => {
    expect(scoreColor(0.7)).toBe("#CA8A04");
    expect(scoreColor(0.51)).toBe("#CA8A04");
  });

  it("low scores are red", () => {
    expect(scoreColor(0.3)).toBe("#DC2626");
    expect(scoreColor(0.0)).toBe("#DC2626");
  });

  it("boundary at 0.8 stays amber (strict >)", () => {
    expect(scoreColor(0.8)).toBe("#CA8A04");
  });

  it("boundary at 0.5 drops to red (strict >)", () => {
    expect(scoreColor(0.5)).toBe("#DC2626");
  });
});
