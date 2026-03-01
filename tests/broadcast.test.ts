import { describe, it, expect } from "vitest";
import * as elementwise from "../src/backend/workers/kernels/elementwise";

function rowMajorStrides(shape: number[]): number[] {
  const strides = new Array(shape.length);
  let s = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = s;
    s *= shape[i];
  }
  return strides;
}

function broadcastStrides(shape: number[], outShape: number[]): number[] {
  const ndim = outShape.length;
  const ndimIn = shape.length;
  const strides = rowMajorStrides(shape);
  const out = new Array(ndim).fill(0);
  for (let i = 0; i < ndim; i++) {
    const dimIn = i - (ndim - ndimIn);
    if (dimIn >= 0 && shape[dimIn] !== 1) {
      out[i] = strides[dimIn];
    }
  }
  return out;
}

describe("elementwise broadcast ops", () => {
  it("add with row broadcast [2,3] + [1,3]", () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]); // [2,3]
    const b = new Float32Array([10, 20, 30]); // [1,3]
    const outShape = [2, 3];
    const stridesA = broadcastStrides([2, 3], outShape);
    const stridesB = broadcastStrides([1, 3], outShape);
    const out = new Float32Array(6);

    elementwise.add(a, b, out, 0, 6, outShape, stridesA, stridesB);

    expect(Array.from(out)).toEqual([11, 22, 33, 14, 25, 36]);
  });

  it("mul with column broadcast [3,1] * [3,4]", () => {
    const a = new Float32Array([2, 3, 4]); // [3,1]
    const b = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]); // [3,4]
    const outShape = [3, 4];
    const stridesA = broadcastStrides([3, 1], outShape);
    const stridesB = broadcastStrides([3, 4], outShape);
    const out = new Float32Array(12);

    elementwise.mul(a, b, out, 0, 12, outShape, stridesA, stridesB);

    const expected = [2, 4, 6, 8, 15, 18, 21, 24, 36, 40, 44, 48];
    expect(Array.from(out)).toEqual(expected);
  });

  it("div with scalar broadcast [2,2] / [1,1]", () => {
    const a = new Float32Array([10, 20, 30, 40]); // [2,2]
    const b = new Float32Array([5]); // [1,1]
    const outShape = [2, 2];
    const stridesA = broadcastStrides([2, 2], outShape);
    const stridesB = broadcastStrides([1, 1], outShape);
    const out = new Float32Array(4);

    elementwise.div(a, b, out, 0, 4, outShape, stridesA, stridesB);

    expect(Array.from(out)).toEqual([2, 4, 6, 8]);
  });
});
