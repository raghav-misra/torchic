import { Tensor, init, shutdown, nn } from "../../src/index";
import { defineTest } from "../framework/define";
import type { RunContext, TestResult } from "../framework/types";

type Backend = "workers" | "wasm" | "webgpu";

function nearlyEqual(a: Float32Array, b: Float32Array, tol: number): { ok: boolean; maxErr: number } {
  if (a.length !== b.length) return { ok: false, maxErr: Infinity };
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return { ok: m <= tol, maxErr: m };
}

function arange(n: number, start = 0, step = 1): number[] {
  return Array.from({ length: n }, (_, i) => start + i * step);
}

async function runShapeOps(backend: Backend, { log }: RunContext): Promise<TestResult> {
  const threadCount = backend === "webgpu" ? 1 : 4;
  await init({ backend, threadCount, memorySizeMB: 32 });
  try {
    const checks: { name: string; ok: boolean; detail: string }[] = [];
    const record = (n: string, ok: boolean, d = "") => {
      checks.push({ name: n, ok, detail: d });
      log(`  ${ok ? "OK " : "FAIL"}  ${n}${d ? "  " + d : ""}`);
    };

    // Concat along last axis (contiguous slabs).
    {
      const a = Tensor.fromData(arange(6), [2, 3]); // 0..5
      const b = Tensor.fromData(arange(4, 100), [2, 2]); // 100..103
      const c = Tensor.concat([a, b], -1);
      const got = await c.toArray();
      const want = new Float32Array([0, 1, 2, 100, 101, 0 + 3, 3 + 1, 3 + 2, 100 + 2, 100 + 3]);
      // want[0..2] = [0,1,2], want[3..4] = [100,101], want[5..7] = [3,4,5], want[8..9] = [102,103]
      const expected = new Float32Array([0, 1, 2, 100, 101, 3, 4, 5, 102, 103]);
      const cmp = nearlyEqual(got, expected, 0);
      record(`concat axis=-1 (contig slabs)`, cmp.ok && c.shape.join(",") === "2,5", `shape=${c.shape} maxErr=${cmp.maxErr}`);
      void want;
    }

    // Concat along axis=1 (channels), 3-D input.
    {
      const a = Tensor.fromData(arange(2 * 3 * 4), [2, 3, 4]);
      const b = Tensor.fromData(arange(2 * 2 * 4, 100), [2, 2, 4]);
      const c = Tensor.concat([a, b], 1);
      const got = await c.toArray();
      const expected = new Float32Array(2 * 5 * 4);
      let k = 0;
      for (let bi = 0; bi < 2; bi++) {
        for (let ci = 0; ci < 3; ci++) {
          for (let li = 0; li < 4; li++) expected[k++] = bi * 12 + ci * 4 + li;
        }
        for (let ci = 0; ci < 2; ci++) {
          for (let li = 0; li < 4; li++) expected[k++] = 100 + bi * 8 + ci * 4 + li;
        }
      }
      const cmp = nearlyEqual(got, expected, 0);
      record(`concat axis=1 [2,3,4]+[2,2,4]`, cmp.ok && c.shape.join(",") === "2,5,4", `shape=${c.shape} maxErr=${cmp.maxErr}`);
    }

    // Concat along axis=0 (outer).
    {
      const a = Tensor.fromData(arange(2 * 3), [2, 3]);
      const b = Tensor.fromData(arange(1 * 3, 100), [1, 3]);
      const c = Tensor.concat([a, b], 0);
      const got = await c.toArray();
      const expected = new Float32Array([0, 1, 2, 3, 4, 5, 100, 101, 102]);
      const cmp = nearlyEqual(got, expected, 0);
      record(`concat axis=0 [2,3]+[1,3]`, cmp.ok && c.shape.join(",") === "3,3", `shape=${c.shape}`);
    }

    // Pad1d — zero pad on last dim.
    {
      const a = Tensor.fromData(arange(6), [2, 3]);
      const p = a.pad1d(1, 2);
      const got = await p.toArray();
      const expected = new Float32Array([0, 0, 1, 2, 0, 0, 0, 3, 4, 5, 0, 0]);
      const cmp = nearlyEqual(got, expected, 0);
      record(`pad1d(1,2) [2,3]`, cmp.ok && p.shape.join(",") === "2,6", `shape=${p.shape}`);
    }

    // Split — 4 chunks along last dim.
    {
      const a = Tensor.fromData(arange(8), [2, 4]);
      const parts = a.split(4, -1);
      const g0 = await parts[0].toArray();
      const g3 = await parts[3].toArray();
      const okShapes = parts.length === 4 && parts.every((t) => t.shape.join(",") === "2,1");
      record(`split(4, -1) [2,4]`, okShapes && g0[0] === 0 && g0[1] === 4 && g3[0] === 3 && g3[1] === 7);
    }

    // GroupNorm — check output has zero mean, unit variance per group.
    {
      const gn = new nn.GroupNorm(2, 8);
      const x = Tensor.fromData(
        Array.from({ length: 2 * 8 * 4 }, (_, i) => Math.sin(i * 0.13)),
        [2, 8, 4],
      );
      const y = await gn.forward(x).toArray();
      // Verify per-(sample,group) normalization: mean ≈ 0, var ≈ 1 across (Cg, L).
      const B = 2;
      const G = 2;
      const Cg = 4;
      const L = 4;
      let maxMeanAbs = 0;
      let maxVarErr = 0;
      for (let b = 0; b < B; b++) {
        for (let g = 0; g < G; g++) {
          let sum = 0;
          const n = Cg * L;
          const base = ((b * G + g) * Cg) * L;
          for (let c = 0; c < Cg; c++) {
            for (let l = 0; l < L; l++) sum += y[base + c * L + l];
          }
          const mean = sum / n;
          let sq = 0;
          for (let c = 0; c < Cg; c++) {
            for (let l = 0; l < L; l++) {
              const d = y[base + c * L + l] - mean;
              sq += d * d;
            }
          }
          const varv = sq / n;
          if (Math.abs(mean) > maxMeanAbs) maxMeanAbs = Math.abs(mean);
          if (Math.abs(varv - 1) > maxVarErr) maxVarErr = Math.abs(varv - 1);
        }
      }
      record(`GroupNorm(2,8) [2,8,4]`, maxMeanAbs < 1e-4 && maxVarErr < 1e-3, `mean|${maxMeanAbs.toExponential(2)} var-1|${maxVarErr.toExponential(2)}`);
    }

    // InstanceNorm1d — same idea, each (b, c) independently normalized.
    {
      const inm = new nn.InstanceNorm1d(4);
      const x = Tensor.fromData(
        Array.from({ length: 3 * 4 * 6 }, (_, i) => Math.cos(i * 0.17)),
        [3, 4, 6],
      );
      const y = await inm.forward(x).toArray();
      let maxMeanAbs = 0;
      let maxVarErr = 0;
      for (let b = 0; b < 3; b++) {
        for (let c = 0; c < 4; c++) {
          const base = (b * 4 + c) * 6;
          let s = 0;
          for (let i = 0; i < 6; i++) s += y[base + i];
          const mean = s / 6;
          let sq = 0;
          for (let i = 0; i < 6; i++) sq += (y[base + i] - mean) * (y[base + i] - mean);
          const varv = sq / 6;
          if (Math.abs(mean) > maxMeanAbs) maxMeanAbs = Math.abs(mean);
          if (Math.abs(varv - 1) > maxVarErr) maxVarErr = Math.abs(varv - 1);
        }
      }
      record(`InstanceNorm1d(4) [3,4,6]`, maxMeanAbs < 1e-4 && maxVarErr < 5e-3, `mean|${maxMeanAbs.toExponential(2)} var-1|${maxVarErr.toExponential(2)}`);
    }

    const allOk = checks.every((c) => c.ok);
    const passed = checks.filter((c) => c.ok).length;
    return {
      pass: allOk,
      message: allOk ? `${passed}/${checks.length} checks passed` : "some checks failed",
    };
  } finally {
    shutdown();
  }
}

defineTest<Backend>({
  name: "Concat + Pad + Split + GroupNorm + InstanceNorm: cross-backend",
  paramName: "backend",
  params: ["workers", "wasm", "webgpu"],
  description:
    "Shape-manipulation ops (Concat, Pad1d, Split) and normalization modules (GroupNorm, InstanceNorm1d) across all three backends.",
  runner: runShapeOps,
});
