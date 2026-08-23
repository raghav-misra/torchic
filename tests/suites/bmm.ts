import { Tensor, init, shutdown } from "../../src/index";
import { defineTest } from "../framework/define";
import type { RunContext, TestResult } from "../framework/types";

type Backend = "workers" | "wasm" | "webgpu";

// Deterministic float fill so cross-backend diffs are apples-to-apples.
function fill(n: number, seed: number): number[] {
  const out: number[] = [];
  let x = seed;
  for (let i = 0; i < n; i++) {
    x = (Math.imul(1103515245, x) + 12345) | 0;
    out.push(((x & 0xffff) - 0x8000) / 0x8000);
  }
  return out;
}

function bmmReference(A: number[], B: number[], bc: number, m: number, k: number, n: number): number[] {
  const out = new Array(bc * m * n).fill(0);
  for (let b = 0; b < bc; b++) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let p = 0; p < k; p++) {
          s += A[b * m * k + i * k + p] * B[b * k * n + p * n + j];
        }
        out[b * m * n + i * n + j] = s;
      }
    }
  }
  return out;
}

async function runBmm(backend: Backend, { log }: RunContext): Promise<TestResult> {
  const threads = backend === "webgpu" ? 1 : 4;
  await init({ backend, threadCount: threads, memorySizeMB: 32 });
  try {
    const cases: Array<{ bc: number; m: number; k: number; n: number }> = [
      { bc: 1, m: 16, k: 16, n: 16 },   // single-batch equivalence
      { bc: 4, m: 8, k: 12, n: 16 },    // small batched
      { bc: 8, m: 32, k: 16, n: 32 },   // typical MHA head shape
      { bc: 3, m: 5, k: 7, n: 11 },     // odd sizes
    ];

    let allPass = true;
    for (const c of cases) {
      const A = fill(c.bc * c.m * c.k, c.bc * 17 + c.m);
      const B = fill(c.bc * c.k * c.n, c.bc * 31 + c.n);
      const at = Tensor.fromData(A, [c.bc, c.m, c.k]);
      const bt = Tensor.fromData(B, [c.bc, c.k, c.n]);

      const out = await at.bmm(bt).toArray();
      const ref = bmmReference(A, B, c.bc, c.m, c.k, c.n);

      let maxErr = 0;
      for (let i = 0; i < out.length; i++) {
        const err = Math.abs(out[i] - ref[i]);
        if (err > maxErr) maxErr = err;
      }
      const ok = maxErr < 1e-3;
      allPass = allPass && ok;
      log(
        `  ${ok ? "OK  " : "FAIL"} bc=${c.bc} m=${c.m} k=${c.k} n=${c.n}  maxErr=${maxErr.toExponential(2)}`,
      );
    }

    // Autograd sanity: bmm forward + backward + shape checks.
    const bc = 3, m = 4, k = 6, n = 5;
    const A = Tensor.fromData(fill(bc * m * k, 111), [bc, m, k], true);
    const B = Tensor.fromData(fill(bc * k * n, 222), [bc, k, n], true);
    const C = A.bmm(B);
    const loss = C.mul(C).sum();
    loss.backward();
    const gA = await A.grad!.toArray();
    const gB = await B.grad!.toArray();
    const gradOk =
      gA.length === bc * m * k &&
      gB.length === bc * k * n &&
      Array.from(gA).every((v) => Number.isFinite(v)) &&
      Array.from(gB).every((v) => Number.isFinite(v)) &&
      Array.from(gA).some((v) => Math.abs(v) > 1e-6) &&
      Array.from(gB).some((v) => Math.abs(v) > 1e-6);
    log(`  ${gradOk ? "OK  " : "FAIL"} autograd: dA size=${gA.length} dB size=${gB.length}`);
    allPass = allPass && gradOk;

    return {
      pass: allPass,
      message: allPass ? `${cases.length + 1}/${cases.length + 1} cases pass` : "some cases failed",
    };
  } finally {
    shutdown();
  }
}

defineTest<Backend>({
  name: "bmm: cross-backend correctness",
  paramName: "backend",
  params: ["workers", "wasm", "webgpu"],
  description:
    "Batched matmul forward + autograd on all three backends, checked against a scalar JS reference.",
  runner: runBmm,
});
