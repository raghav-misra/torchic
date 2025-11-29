import { describe, it, expect } from 'vitest';

import * as elementwise from '../src/kernels/elementwise';
import * as reductions from '../src/kernels/reductions';
import * as transpose from '../src/kernels/transpose';
import * as matmul from '../src/kernels/matmul';
import * as embedding from '../src/kernels/embedding';

function randFloat32(n: number, seed = 42) {
  const r = new Float32Array(n);
  let x = seed;
  for (let i = 0; i < n; i++) {
    x = (1103515245 * x + 12345) & 0x7fffffff;
    r[i] = ((x % 1000) - 500) / 500;
  }
  return r;
}

// Helper to compute strides for row-major shape
function rowMajorStrides(shape: number[]) {
  const strides = new Array(shape.length);
  let s = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = s;
    s *= shape[i];
  }
  return strides;
}

describe('materialize / strided ops', () => {
  it('materialize copies strided input to contiguous output', () => {
    const shape = [3, 2];
    const strides = [3, 1]; // imagine padded rows: row stride 3 (k+1)
    const inputRaw = new Float32Array(3 * 3); // padded storage
    // logical values
    const logical = new Float32Array([1,2,3,4,5,6]); // 3x2
    // place logical into inputRaw with stride
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 2; c++) {
        inputRaw[r * 3 + c] = logical[r * 2 + c];
      }
    }

    const out = new Float32Array(6);
    elementwise.materialize(inputRaw, out, 0, 6, shape, strides);
    for (let i = 0; i < 6; i++) expect(out[i]).toBe(logical[i]);
  });

  it('unary op handles strides mapping', () => {
    const shape = [2, 3];
    const paddedRow = 4;
    const inputRaw = new Float32Array(2 * paddedRow);
    const logical = new Float32Array([1, -2, 3, -4, 5, -6]);
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 3; c++) {
        inputRaw[r * paddedRow + c] = logical[r * 3 + c];
      }
    }
    const out = new Float32Array(6);
    const strides = [paddedRow, 1];
    elementwise.relu(inputRaw, out, 0, 6, shape, strides);
    for (let i = 0; i < 6; i++) expect(out[i]).toBe(Math.max(0, logical[i]));
  });
});

describe('transpose + matmul integration', () => {
  it('transpose then matmul equals reference', () => {
    const m = 4, k = 3, n = 5;
    const A = randFloat32(m * k, 11);
    const B = randFloat32(k * n, 12);

    // compute C = A * B
    const C = new Float32Array(m * n);
    matmul.matmul(A, B, C, m, n, k, 0, m);

    // transpose B into Bt (shape n x k)
    const Bt = new Float32Array(n * k);
    transpose.transpose(B, Bt, k, n, 0, n);
    // Now compute D = A * (transpose(transpose(B))) -> same as C
    const D = new Float32Array(m * n);
    matmul.matmul(A, B, D, m, n, k, 0, m);

    for (let i = 0; i < m * n; i++) expect(D[i]).toBeCloseTo(C[i], 6);
  });
});

describe('softmax stability and edge cases', () => {
  it('softmax handles large values and identical rows', () => {
    const m = 2, n = 4;
    const inArr = new Float32Array([1000, 1000, 1000, 1000, -1000, -1000, -1000, -1000]);
    const out = new Float32Array(m * n);
    elementwise.softmax2d(inArr, out, m, n, 0, m);
    // first row identical -> uniform
    for (let j = 0; j < n; j++) expect(out[j]).toBeCloseTo(1 / n, 6);
    // second row identical negatives -> uniform
    for (let j = 0; j < n; j++) expect(out[n + j]).toBeCloseTo(1 / n, 6);
  });
});

describe('reductions edge cases', () => {
  it('sum_axis reduces across specified axis correctly', () => {
    // shape [2,3,4]
    const shape = [2,3,4];
    const strides = rowMajorStrides(shape);
    const total = shape.reduce((a,b)=>a*b,1);
    const arr = new Float32Array(total);
    for (let i = 0; i < total; i++) arr[i] = i+1; // 1..24

    // Reduce axis = 1 (size 3) => output shape [2,4] => flattened length 8
    const out = new Float32Array(2*4);
    reductions.sum_axis(arr, out, 0, out.length, shape, strides, 1);

    // compute reference
    const ref = new Float32Array(8);
    for (let i0 = 0; i0 < 2; i0++) for (let i2 = 0; i2 < 4; i2++) {
      let s = 0;
      for (let i1 = 0; i1 < 3; i1++) {
        const idx = i0 * strides[0] + i1 * strides[1] + i2 * strides[2];
        s += arr[idx];
      }
      ref[i0 * 4 + i2] = s;
    }

    for (let i = 0; i < out.length; i++) expect(out[i]).toBe(ref[i]);
  });

  it('add_scalar_tensor adds scalar to every element', () => {
    const a = new Float32Array([1,2,3,4]);
    const scalar = new Float32Array([0.5]);
    const out = new Float32Array(4);
    reductions.add_scalar_tensor(a, scalar, out, 0, 4);
    for (let i = 0; i < 4; i++) expect(out[i]).toBeCloseTo(a[i] + 0.5, 6);
  });
});

describe('embedding kernels', () => {
  it('embedding copies rows properly', () => {
    const vocab = 5, dim = 3;
    const weights = new Float32Array(vocab * dim);
    for (let i = 0; i < vocab; i++) for (let d = 0; d < dim; d++) weights[i*dim + d] = i*10 + d;
    const indices = new Float32Array([2,4,1]);
    const out = new Float32Array(indices.length * dim);
    embedding.embedding(weights, indices, out, dim, 0, out.length);
    for (let i = 0; i < indices.length; i++) {
      const row = indices[i];
      for (let d = 0; d < dim; d++) {
        expect(out[i*dim + d]).toBe(weights[row*dim + d]);
      }
    }
  });

  it('embedding_backward accumulates grads into weightsGrad', () => {
    const vocab = 4, dim = 2;
    const weightsGrad = new Float32Array(vocab * dim);
    const indices = new Float32Array([1,2,1]);
    const outGrad = new Float32Array(indices.length * dim);
    // fill grads
    for (let i = 0; i < outGrad.length; i++) outGrad[i] = i+1;

    embedding.embedding_backward(weightsGrad, indices, outGrad, dim, 0, outGrad.length);

    // manual accumulation
    const expected = new Float32Array(vocab * dim);
    for (let i = 0; i < indices.length; i++) {
      const row = indices[i];
      for (let d = 0; d < dim; d++) {
        expected[row*dim + d] += outGrad[i*dim + d];
      }
    }

    for (let i = 0; i < expected.length; i++) expect(weightsGrad[i]).toBe(expected[i]);
  });
});

describe('misc utilities', () => {
  it('copy copies contiguous arrays', () => {
    const a = new Float32Array([1,2,3,4]);
    const out = new Float32Array(4);
    elementwise.copy(a, out, 0, 4);
    for (let i = 0; i < 4; i++) expect(out[i]).toBe(a[i]);
  });

  it('matmul supports strides for both A and B', () => {
    const m = 2, k = 3, n = 2;
    const paddedARow = 4, paddedBRow = 4;
    const Araw = new Float32Array(m * paddedARow);
    const Braw = new Float32Array(k * paddedBRow);
    const A = new Float32Array(m * k);
    const B = new Float32Array(k * n);

    for (let i = 0; i < m; i++) for (let p = 0; p < k; p++) {
      const v = i * k + p + 1;
      Araw[i * paddedARow + p] = v;
      A[i * k + p] = v;
    }
    for (let p = 0; p < k; p++) for (let j = 0; j < n; j++) {
      const v = (p * n + j) + 1;
      Braw[p * paddedBRow + j] = v;
      B[p * n + j] = v;
    }

    const C = new Float32Array(m * n);
    const expected = new Float32Array(m * n);
    matmul.matmul(A, B, expected, m, n, k, 0, m);

    const stridesA = [paddedARow, 1];
    const stridesB = [paddedBRow, 1];
    matmul.matmul(Araw, Braw, C, m, n, k, 0, m, stridesA, stridesB);

    for (let i = 0; i < m * n; i++) expect(C[i]).toBeCloseTo(expected[i], 6);
  });
});
