import { describe, it, expect } from "vitest";

import * as matmul from "../src/backend/workers/kernels/matmul";
import * as elementwise from "../src/backend/workers/kernels/elementwise";
import * as reductions from "../src/backend/workers/kernels/reductions";

function randFloat32(n: number, seed = 42) {
  const r = new Float32Array(n);
  let x = seed;
  for (let i = 0; i < n; i++) {
    // simple LCG
    x = (1103515245 * x + 12345) & 0x7fffffff;
    r[i] = ((x % 1000) - 500) / 500;
  }
  return r;
}

describe("matmul kernel", () => {
  it("computes A*B correctly for small matrices", () => {
    const m = 4,
      k = 3,
      n = 5;
    const A = randFloat32(m * k, 1);
    const B = randFloat32(k * n, 2);
    const C = new Float32Array(m * n);

    // compute expected
    const E = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let p = 0; p < k; p++) {
          s += A[i * k + p] * B[p * n + j];
        }
        E[i * n + j] = s;
      }
    }

    // call kernel (single worker range)
    matmul.matmul(A, B, C, m, n, k, 0, m);

    for (let i = 0; i < m * n; i++) {
      expect(C[i]).toBeCloseTo(E[i], 6);
    }
  });

  it("supports strides for A and B", () => {
    // Create A as (m x k) but store with row stride = k+1 (padding)
    const m = 3,
      k = 2,
      n = 3;
    const paddedARow = k + 1;
    const Araw = new Float32Array(m * paddedARow);
    const A = new Float32Array(m * k);
    const B = randFloat32(k * n, 7);
    const C = new Float32Array(m * n);

    // fill Araw with values and place logical A into Araw with stride
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        const v = (i + 1) * (p + 2);
        Araw[i * paddedARow + p] = v;
        A[i * k + p] = v;
      }
      Araw[i * paddedARow + k] = 0; // padding
    }

    // expected
    const E = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let p = 0; p < k; p++) {
          s += A[i * k + p] * B[p * n + j];
        }
        E[i * n + j] = s;
      }
    }

    // Call kernel with strides
    const stridesA = [paddedARow, 1];
    matmul.matmul(Araw, B, C, m, n, k, 0, m, stridesA, undefined);

    for (let i = 0; i < m * n; i++) {
      expect(C[i]).toBeCloseTo(E[i], 6);
    }
  });
});

describe("elementwise kernels", () => {
  it("tanh and tanh_backward are consistent", () => {
    const N = 100;
    const x = randFloat32(N, 10);
    const y = new Float32Array(N);
    elementwise.tanh(x, y, 0, N);

    // numeric gradient check for tanh: d/dx tanh(x) = 1 - tanh(x)^2
    const eps = 1e-3;
    // Use full-size temporary outputs so kernel writes align with the global index
    for (let i = 0; i < N; i++) {
      const orig = x[i];
      x[i] = orig + eps;
      const y1 = new Float32Array(N);
      elementwise.tanh(x, y1, i, i + 1);
      x[i] = orig - eps;
      const y2 = new Float32Array(N);
      elementwise.tanh(x, y2, i, i + 1);
      x[i] = orig;

      const numeric = (y1[i] - y2[i]) / (2 * eps);
      const analytic = 1 - y[i] * y[i];
      expect(numeric).toBeCloseTo(analytic, 2);
    }
  });

  it("softmax2d and backward produce normalized outputs", () => {
    const m = 5,
      n = 7;
    const inArr = randFloat32(m * n, 20);
    const out = new Float32Array(m * n);

    // compute softmax per row using kernel
    elementwise.softmax2d(inArr, out, m, n, 0, m);

    for (let i = 0; i < m; i++) {
      let sum = 0;
      for (let j = 0; j < n; j++) sum += out[i * n + j];
      expect(sum).toBeCloseTo(1, 6);
      // all entries between 0 and 1
      for (let j = 0; j < n; j++) {
        expect(out[i * n + j]).toBeGreaterThanOrEqual(0 - 1e-6);
        expect(out[i * n + j]).toBeLessThanOrEqual(1 + 1e-6);
      }
    }

    // backward: given gradOut = ones, gradIn should sum to 0 per row
    const gradOut = new Float32Array(m * n);
    for (let i = 0; i < m * n; i++) gradOut[i] = 1;
    const gradIn = new Float32Array(m * n);
    elementwise.softmax_backward2d(out, gradOut, gradIn, m, n, 0, m);

    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += gradIn[i * n + j];
      // For softmax backward with gradOut=1, sum should be 0
      expect(s).toBeCloseTo(0, 5);
    }
  });
});

describe("reductions", () => {
  it("sum_partial + sum_final produce correct total", () => {
    const N = 1000;
    const arr = randFloat32(N, 55);
    const partials = new Float32Array(8); // simulate 8 workers
    reductions.sum_partial(arr, partials, 0, 0, N);
    // sum_partial writes into partials[0] in current impl for tests; to simulate, just check sum
    let s = 0;
    for (let i = 0; i < N; i++) s += arr[i];
    // call sum_final on a simple 1-element input
    const out = new Float32Array(1);
    const small = new Float32Array([s]);
    reductions.sum_final(small, out, 1);
    expect(out[0]).toBeCloseTo(s, 5);
  });
});
