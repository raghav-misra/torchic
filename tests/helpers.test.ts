import { describe, it, expect } from "vitest";
import { inferShape, countElements, flattenInto } from "../src/frontend/tensor";
import type { NestedArray } from "../src/frontend/tensor";

describe("fromData helpers", () => {
  it("inferShape on nested 2D array", () => {
    const data: NestedArray = [
      [1, 2],
      [3, 4],
      [5, 6],
    ];
    expect(inferShape(data)).toEqual([3, 2]);
  });

  it("inferShape on Float32Array", () => {
    expect(inferShape(new Float32Array([1, 2, 3]))).toEqual([3]);
  });

  it("countElements + flattenInto round-trip on 3D array", () => {
    const data: NestedArray = [
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
      [
        [9, 10],
        [11, 12],
      ],
    ];

    const shape = inferShape(data);
    expect(shape).toEqual([3, 2, 2]);

    const n = countElements(data);
    expect(n).toBe(3 * 2 * 2);

    const out = new Float32Array(n);
    flattenInto(data, out, 0);

    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  });
});
