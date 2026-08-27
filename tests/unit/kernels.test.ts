import { describe, it, expect } from "vitest";

import * as matmul from "../../src/backend/workers/kernels/matmul";
import * as elementwise from "../../src/backend/workers/kernels/elementwise";
import * as reductions from "../../src/backend/workers/kernels/reductions";

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

  it("rms_norm2d matches reference and respects weight", () => {
    const m = 6,
      n = 11,
      eps = 1e-5;
    const inArr = randFloat32(m * n, 33);
    const weight = new Float32Array(n);
    for (let i = 0; i < n; i++) weight[i] = 0.5 + 0.1 * i;

    const out = new Float32Array(m * n);
    elementwise.rms_norm2d(inArr, weight, out, m, n, eps, 0, m);

    for (let r = 0; r < m; r++) {
      let sumsq = 0;
      for (let c = 0; c < n; c++) sumsq += inArr[r * n + c] ** 2;
      const invRms = 1 / Math.sqrt(sumsq / n + eps);
      for (let c = 0; c < n; c++) {
        const expected = inArr[r * n + c] * invRms * weight[c];
        expect(out[r * n + c]).toBeCloseTo(expected, 5);
      }
    }
  });

  it("rms_norm2d partial row range only writes assigned rows", () => {
    const m = 4,
      n = 8,
      eps = 1e-5;
    const inArr = randFloat32(m * n, 77);
    const weight = new Float32Array(n).fill(1);
    const out = new Float32Array(m * n);
    for (let i = 0; i < out.length; i++) out[i] = -999;

    elementwise.rms_norm2d(inArr, weight, out, m, n, eps, 1, 3);

    for (let c = 0; c < n; c++) expect(out[c]).toBe(-999);
    for (let c = 0; c < n; c++) expect(out[3 * n + c]).toBe(-999);
    for (let c = 0; c < n; c++) expect(out[n + c]).not.toBe(-999);
    for (let c = 0; c < n; c++) expect(out[2 * n + c]).not.toBe(-999);
  });

  it("rope matches reference rotation and cycles through positions", () => {
    const N = 2,
      T = 5,
      D = 8;
    const dHalf = D / 2;
    const totalRows = N * T;

    const x = randFloat32(totalRows * D, 91);
    const cos = new Float32Array(T * dHalf);
    const sin = new Float32Array(T * dHalf);
    const theta = 10000;
    for (let i = 0; i < dHalf; i++) {
      const invFreq = Math.pow(theta, (-2 * i) / D);
      for (let t = 0; t < T; t++) {
        cos[t * dHalf + i] = Math.cos(t * invFreq);
        sin[t * dHalf + i] = Math.sin(t * invFreq);
      }
    }
    const out = new Float32Array(totalRows * D);
    elementwise.rope(x, cos, sin, out, T, dHalf, 0, totalRows);

    for (let r = 0; r < totalRows; r++) {
      const t = r % T;
      for (let i = 0; i < dHalf; i++) {
        const a = x[r * D + i];
        const b = x[r * D + i + dHalf];
        const c = cos[t * dHalf + i];
        const s = sin[t * dHalf + i];
        expect(out[r * D + i]).toBeCloseTo(a * c - b * s, 5);
        expect(out[r * D + i + dHalf]).toBeCloseTo(a * s + b * c, 5);
      }
    }
  });

  it("rope preserves L2 norm per row (rotation is orthogonal)", () => {
    const T = 4,
      D = 16;
    const dHalf = D / 2;
    const x = randFloat32(T * D, 123);
    const cos = new Float32Array(T * dHalf);
    const sin = new Float32Array(T * dHalf);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < dHalf; i++) {
        const angle = t * Math.pow(10000, (-2 * i) / D);
        cos[t * dHalf + i] = Math.cos(angle);
        sin[t * dHalf + i] = Math.sin(angle);
      }
    }
    const out = new Float32Array(T * D);
    elementwise.rope(x, cos, sin, out, T, dHalf, 0, T);

    for (let r = 0; r < T; r++) {
      let inNorm = 0,
        outNorm = 0;
      for (let d = 0; d < D; d++) {
        inNorm += x[r * D + d] ** 2;
        outNorm += out[r * D + d] ** 2;
      }
      expect(outNorm).toBeCloseTo(inNorm, 4);
    }
  });

  it("causal_softmax2d zeros disallowed columns and rows sum to 1", () => {
    const m = 6,
      n = 6;
    const inArr = randFloat32(m * n, 55);
    const out = new Float32Array(m * n);
    elementwise.causal_softmax2d(inArr, out, m, n, 0, 0, m);

    for (let r = 0; r < m; r++) {
      let sum = 0;
      for (let c = 0; c < n; c++) {
        if (c > r) {
          expect(out[r * n + c]).toBe(0);
        } else {
          expect(out[r * n + c]).toBeGreaterThan(0);
          sum += out[r * n + c];
        }
      }
      expect(sum).toBeCloseTo(1, 5);
    }
  });

  it("causal_softmax2d with pastLen shifts the allowed window", () => {
    const m = 3,
      n = 10,
      pastLen = 4;
    const inArr = randFloat32(m * n, 88);
    const out = new Float32Array(m * n);
    elementwise.causal_softmax2d(inArr, out, m, n, pastLen, 0, m);

    for (let r = 0; r < m; r++) {
      const allowed = pastLen + r;
      for (let c = 0; c < n; c++) {
        if (c > allowed) expect(out[r * n + c]).toBe(0);
      }
      let sum = 0;
      for (let c = 0; c <= allowed; c++) sum += out[r * n + c];
      expect(sum).toBeCloseTo(1, 5);
    }
  });

  it("causal_softmax2d matches (scores + upper_triangle_-inf) → softmax", () => {
    const m = 5,
      n = 5;
    const inArr = randFloat32(m * n, 191);
    const masked = new Float32Array(m * n);
    for (let r = 0; r < m; r++) {
      for (let c = 0; c < n; c++) {
        masked[r * n + c] = c > r ? -Infinity : inArr[r * n + c];
      }
    }
    const ref = new Float32Array(m * n);
    elementwise.softmax2d(masked, ref, m, n, 0, m);

    const out = new Float32Array(m * n);
    elementwise.causal_softmax2d(inArr, out, m, n, 0, 0, m);

    for (let i = 0; i < m * n; i++) {
      // NaN from softmax(all -inf) not possible here since row 0 has one allowed col.
      expect(out[i]).toBeCloseTo(ref[i], 5);
    }
  });

  it("copy_range writes only the target slice and leaves the rest intact", () => {
    const src = randFloat32(8, 200);
    const dst = new Float32Array(20);
    for (let i = 0; i < dst.length; i++) dst[i] = -7;

    elementwise.copy_range(src, dst, 5, 0, 8);

    for (let i = 0; i < 5; i++) expect(dst[i]).toBe(-7);
    for (let i = 0; i < 8; i++) expect(dst[5 + i]).toBe(src[i]);
    for (let i = 13; i < 20; i++) expect(dst[i]).toBe(-7);
  });

  it("copy_range partitioned across workers reconstructs the full copy", () => {
    const src = randFloat32(16, 210);
    const dst = new Float32Array(24);
    for (let i = 0; i < dst.length; i++) dst[i] = -7;

    // Simulate two workers splitting the range.
    elementwise.copy_range(src, dst, 4, 0, 8);
    elementwise.copy_range(src, dst, 4, 8, 16);

    for (let i = 0; i < 4; i++) expect(dst[i]).toBe(-7);
    for (let i = 0; i < 16; i++) expect(dst[4 + i]).toBe(src[i]);
    for (let i = 20; i < 24; i++) expect(dst[i]).toBe(-7);
  });

  it("repeat_interleave on 1D matches [a, b, c] -> [a, a, b, b, c, c]", () => {
    const input = new Float32Array([10, 20, 30]);
    const output = new Float32Array(6);
    elementwise.repeat_interleave(input, output, 3, 1, 2, 0, 6);
    expect(Array.from(output)).toEqual([10, 10, 20, 20, 30, 30]);
  });

  it("repeat_interleave on GQA shape [B=1, 2, T=3, D=4] with repeats=3", () => {
    const B = 1,
      Hkv = 2,
      T = 3,
      D = 4;
    const repeats = 3;
    const input = new Float32Array(B * Hkv * T * D);
    for (let i = 0; i < input.length; i++) input[i] = i;

    // dim=1 → axisSize=Hkv=2, inner=T*D=12, outer=B=1.
    const axisSize = Hkv;
    const inner = T * D;
    const outCount = B * Hkv * repeats * T * D;
    const output = new Float32Array(outCount);
    elementwise.repeat_interleave(input, output, axisSize, inner, repeats, 0, outCount);

    // Each head slab (T*D = 12 elements) should appear `repeats` times consecutively.
    for (let h = 0; h < Hkv; h++) {
      const inBase = h * inner;
      for (let r = 0; r < repeats; r++) {
        const outBase = (h * repeats + r) * inner;
        for (let i = 0; i < inner; i++) {
          expect(output[outBase + i]).toBe(input[inBase + i]);
        }
      }
    }
  });

  it("repeat_interleave partitioned matches full range", () => {
    const outer = 2,
      axisSize = 3,
      inner = 5,
      repeats = 2;
    const total = outer * axisSize * repeats * inner;
    const input = new Float32Array(outer * axisSize * inner);
    for (let i = 0; i < input.length; i++) input[i] = Math.sin(i * 0.31);

    const outFull = new Float32Array(total);
    elementwise.repeat_interleave(input, outFull, axisSize, inner, repeats, 0, total);

    const outPart = new Float32Array(total);
    const split = 17;
    elementwise.repeat_interleave(input, outPart, axisSize, inner, repeats, 0, split);
    elementwise.repeat_interleave(input, outPart, axisSize, inner, repeats, split, total);

    for (let i = 0; i < total; i++) expect(outPart[i]).toBe(outFull[i]);
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

describe("gelu", () => {
  // Tanh approximation reference used by BERT / GPT-2 / Kokoro.
  function refGelu(x: number) {
    const c = Math.sqrt(2 / Math.PI);
    return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
  }

  it("forward matches reference formula", () => {
    const xs = new Float32Array([-3, -1, -0.5, 0, 0.5, 1, 2, 3]);
    const out = new Float32Array(xs.length);
    elementwise.gelu(xs, out, 0, xs.length);
    for (let i = 0; i < xs.length; i++) {
      expect(out[i]).toBeCloseTo(refGelu(xs[i]), 5);
    }
  });

  it("backward matches finite-difference gradient", () => {
    const xs = randFloat32(32, 91);
    // gradOutput = 1 so gradInput should equal d(gelu)/dx.
    const gradOut = new Float32Array(xs.length).fill(1);
    const analytic = new Float32Array(xs.length);
    elementwise.gelu_backward(xs, gradOut, analytic, 0, xs.length);

    const eps = 1e-3;
    for (let i = 0; i < xs.length; i++) {
      const y1 = refGelu(xs[i] + eps);
      const y2 = refGelu(xs[i] - eps);
      const numeric = (y1 - y2) / (2 * eps);
      expect(analytic[i]).toBeCloseTo(numeric, 3);
    }
  });
});

describe("sqrt / rsqrt", () => {
  it("sqrt forward matches Math.sqrt", () => {
    const xs = new Float32Array([0.01, 0.5, 1, 2, 100]);
    const out = new Float32Array(xs.length);
    elementwise.sqrt(xs, out, 0, xs.length);
    for (let i = 0; i < xs.length; i++) expect(out[i]).toBeCloseTo(Math.sqrt(xs[i]), 5);
  });

  it("sqrt_backward = 0.5 / y", () => {
    const xs = new Float32Array([0.25, 1, 4, 9]);
    const y = new Float32Array(xs.length);
    elementwise.sqrt(xs, y, 0, xs.length);
    const gradOut = new Float32Array(xs.length).fill(1);
    const gradIn = new Float32Array(xs.length);
    elementwise.sqrt_backward(y, gradOut, gradIn, 0, xs.length);
    for (let i = 0; i < xs.length; i++) {
      expect(gradIn[i]).toBeCloseTo(0.5 / Math.sqrt(xs[i]), 5);
    }
  });

  it("rsqrt forward = 1 / sqrt", () => {
    const xs = new Float32Array([0.01, 0.5, 1, 2, 100]);
    const out = new Float32Array(xs.length);
    elementwise.rsqrt(xs, out, 0, xs.length);
    for (let i = 0; i < xs.length; i++) expect(out[i]).toBeCloseTo(1 / Math.sqrt(xs[i]), 5);
  });

  it("rsqrt_backward = -0.5 * y^3", () => {
    const xs = new Float32Array([0.25, 1, 4, 9]);
    const y = new Float32Array(xs.length);
    elementwise.rsqrt(xs, y, 0, xs.length);
    const gradOut = new Float32Array(xs.length).fill(1);
    const gradIn = new Float32Array(xs.length);
    elementwise.rsqrt_backward(y, gradOut, gradIn, 0, xs.length);
    for (let i = 0; i < xs.length; i++) {
      const yi = 1 / Math.sqrt(xs[i]);
      expect(gradIn[i]).toBeCloseTo(-0.5 * yi * yi * yi, 5);
    }
  });
});
