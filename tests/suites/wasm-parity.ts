import { Tensor, init, shutdown } from "../../src/index";
import { defineTest } from "../framework/define";
import type { RunContext } from "../framework/types";

type Backend = "workers" | "wasm";

interface Results {
  add: Float32Array;
  sub: Float32Array;
  mul: Float32Array;
  div: Float32Array;
  neg: Float32Array;
  relu: Float32Array;
  exp: Float32Array;
  log: Float32Array;
  matmul: Float32Array;
  matmul_odd_n: Float32Array;
  broadcast_add_row27: Float32Array;
  matmul_add_chain: Float32Array;
  transpose: Float32Array;
  softmax: Float32Array;
  sum: number;
  sum_axis_last: Float32Array;
  sum_axis_first: Float32Array;
  broadcast_mul_scalar: Float32Array;
  broadcast_add_row: Float32Array;
  gelu: Float32Array;
  sqrt: Float32Array;
  rsqrt: Float32Array;
  sigmoid: Float32Array;
}

async function runOps(backend: Backend, threads: number): Promise<Results> {
  await init({ backend, threadCount: threads, memorySizeMB: 64 });

  const aData = new Float32Array(64 * 64);
  const bData = new Float32Array(64 * 64);
  for (let i = 0; i < aData.length; i++) aData[i] = Math.sin(i * 0.13);
  for (let i = 0; i < bData.length; i++) bData[i] = Math.cos(i * 0.07);

  const a = Tensor.fromData(Array.from(aData), [64, 64]);
  const b = Tensor.fromData(Array.from(bData), [64, 64]);

  const scalar = Tensor.fromData([0.5], [1]);
  const rowVec = Tensor.fromData(
    Array.from({ length: 64 }, (_, i) => Math.sin(i * 0.31)),
    [1, 64],
  );

  // The [256, 200] * [200, 27] shape from makemore's hidden -> logits matmul.
  // n=27 doesn't divide by NR=8, exercises SIMD + scalar-tail path.
  const mmA = Tensor.fromData(
    Array.from({ length: 256 * 200 }, (_, i) => Math.sin(i * 0.017) * 0.01),
    [256, 200],
  );
  const mmB = Tensor.fromData(
    Array.from({ length: 200 * 27 }, (_, i) => Math.cos(i * 0.023) * 0.01),
    [200, 27],
  );

  // Broadcast add of a bias row (shape [1, 27]) into the [256, 27] output.
  // Same pattern as `logits = matmul + bout` in makemore.
  const wide = Tensor.fromData(
    Array.from({ length: 256 * 27 }, (_, i) => Math.sin(i * 0.03) * 0.05),
    [256, 27],
  );
  const rowBias = Tensor.fromData(
    Array.from({ length: 27 }, (_, i) => Math.cos(i * 0.19) * 0.1),
    [1, 27],
  );

  // End-to-end makemore-shaped chain that combines matmul + broadcast add.
  const chain = await mmA.matmul(mmB).add(rowBias).toArray();

  const results = {
    add: await a.add(b).toArray(),
    sub: await a.sub(b).toArray(),
    mul: await a.mul(b).toArray(),
    div: await a.div(b).toArray(),
    neg: await a.neg().toArray(),
    relu: await a.relu().toArray(),
    exp: await a.exp().toArray(),
    log: await a.mul(a).log().toArray(), // avoid log(negative)
    matmul: await a.matmul(b).toArray(),
    matmul_odd_n: await mmA.matmul(mmB).toArray(),
    broadcast_add_row27: await wide.add(rowBias).toArray(),
    matmul_add_chain: chain,
    transpose: await a.transpose().toArray(),
    softmax: await a.softmax(1).toArray(),
    sum: await a.sum().item(),
    sum_axis_last: await a.sum(1).toArray(),
    sum_axis_first: await a.sum(0).toArray(),
    broadcast_mul_scalar: await a.mul(scalar).toArray(),
    broadcast_add_row: await a.add(rowVec).toArray(),
    gelu: await a.gelu().toArray(),
    sqrt: await a.mul(a).sqrt().toArray(), // avoid negative inputs
    rsqrt: await a.mul(a).add(Tensor.fromData([1e-3])).rsqrt().toArray(),
    sigmoid: await a.sigmoid().toArray(),
  };

  shutdown();
  return results;
}

function maxAbsDiff(x: Float32Array | number, y: Float32Array | number): number {
  if (typeof x === "number" && typeof y === "number") return Math.abs(x - y);
  if (typeof x === "number" || typeof y === "number") {
    throw new Error("type mismatch in maxAbsDiff");
  }
  let m = 0;
  for (let i = 0; i < x.length; i++) {
    const d = Math.abs(x[i] - y[i]);
    if (d > m) m = d;
  }
  return m;
}

// Ops where SIMD lane order or fused SIMD reductions may drift within float32 noise.
const TOLERANCES: Record<keyof Results, number> = {
  add: 1e-6,
  sub: 1e-6,
  mul: 1e-6,
  div: 1e-5,
  neg: 0,
  relu: 0,
  exp: 1e-5,
  log: 1e-5,
  matmul: 1e-3,
  matmul_odd_n: 1e-3,
  broadcast_add_row27: 1e-6,
  matmul_add_chain: 1e-3,
  transpose: 0,
  softmax: 1e-6,
  sum: 1e-2,
  sum_axis_last: 1e-5,
  sum_axis_first: 1e-5,
  broadcast_mul_scalar: 1e-6,
  broadcast_add_row: 1e-6,
  gelu: 1e-5,
  sqrt: 1e-5,
  rsqrt: 1e-5,
  sigmoid: 1e-6,
};

async function runParity(threads: number, { log }: RunContext) {
  log(`workers backend, threads=${threads}`);
  const w = await runOps("workers", threads);
  log(`wasm backend, threads=${threads}`);
  const a = await runOps("wasm", threads);

  const rows: string[] = [];
  let allOk = true;
  for (const key of Object.keys(w) as (keyof Results)[]) {
    const diff = maxAbsDiff(w[key], a[key]);
    const tol = TOLERANCES[key];
    const ok = diff <= tol;
    if (!ok) allOk = false;
    rows.push(
      `  ${key.padEnd(10)} max|w-a| = ${diff.toExponential(3).padStart(11)}   tol ${tol.toExponential(1)}   ${ok ? "OK" : "FAIL"}`,
    );
  }
  for (const r of rows) log(r);

  return {
    pass: allOk,
    message: allOk ? `${rows.length}/${rows.length} kernels match` : "some kernels diverged",
  };
}

defineTest({
  name: "WASM ↔ Workers kernel parity",
  paramName: "threads",
  params: [1, 2, 4, 8],
  description: "Runs 12 kernels on both backends with fixed inputs and diffs their outputs.",
  runner: runParity,
});
