import { describe, it, expect } from "vitest";

// Bbox hit-test: the canonical behaviour shared between the canvas picker
// (pixel-accurate, via the label mask) and any bbox-based fallback. The real
// picker uses pixel ids from the mask PNG; this helper verifies the bbox
// contract — first-match-wins, inclusive boundaries — that the rest of the
// UI (e.g. overlapping thumbnails) relies on.
function bboxHit(x, y, objects) {
  for (const o of objects) {
    const [x1, y1, x2, y2] = o.bbox;
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) return o.object_id;
  }
  return null;
}

describe("bbox hit-test", () => {
  it("click inside a cell returns its id", () => {
    expect(bboxHit(30, 30, [{ object_id: 1, bbox: [10, 10, 50, 50] }])).toBe(1);
  });

  it("click outside returns null", () => {
    expect(bboxHit(60, 60, [{ object_id: 1, bbox: [10, 10, 50, 50] }])).toBeNull();
  });

  it("click on a boundary is inclusive", () => {
    const objs = [{ object_id: 1, bbox: [10, 10, 50, 50] }];
    expect(bboxHit(10, 10, objs)).toBe(1);
    expect(bboxHit(50, 50, objs)).toBe(1);
  });

  it("first match wins on overlap", () => {
    const objs = [
      { object_id: 1, bbox: [10, 10, 50, 50] },
      { object_id: 2, bbox: [30, 30, 70, 70] },
    ];
    expect(bboxHit(40, 40, objs)).toBe(1);
  });

  it("multiple non-overlapping cells", () => {
    const objs = [
      { object_id: 1, bbox: [10, 10, 30, 30] },
      { object_id: 2, bbox: [50, 50, 70, 70] },
      { object_id: 3, bbox: [90, 10, 110, 30] },
    ];
    expect(bboxHit(20, 20, objs)).toBe(1);
    expect(bboxHit(60, 60, objs)).toBe(2);
    expect(bboxHit(100, 20, objs)).toBe(3);
    expect(bboxHit(40, 40, objs)).toBeNull();
  });

  it("empty object list returns null", () => {
    expect(bboxHit(50, 50, [])).toBeNull();
  });
});
