import { Tensor, init, shutdown } from "../../src/index";
import { defineTest } from "../framework/define";
import type { RunContext } from "../framework/types";

type Backend = "workers" | "webgpu";

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
  transpose: Float32Array;
  softmax: Float32Array;
  sum: number;
}

async function runOps(backend: Backend): Promise<Results> {
  await init({ backend, memorySizeMB: 64 });

  const aData = new Float32Array(64 * 64);
  const bData = new Float32Array(64 * 64);
  for (let i = 0; i < aData.length; i++) aData[i] = Math.sin(i * 0.13);
  for (let i = 0; i < bData.length; i++) bData[i] = Math.cos(i * 0.07);

  const a = Tensor.fromData(Array.from(aData), [64, 64]);
  const b = Tensor.fromData(Array.from(bData), [64, 64]);

  const results = {
    add: await a.add(b).toArray(),
    sub: await a.sub(b).toArray(),
    mul: await a.mul(b).toArray(),
    div: await a.div(b).toArray(),
    neg: await a.neg().toArray(),
    relu: await a.relu().toArray(),
    exp: await a.exp().toArray(),
    log: await a.mul(a).log().toArray(),
    matmul: await a.matmul(b).toArray(),
    transpose: await a.transpose().toArray(),
    softmax: await a.softmax(1).toArray(),
    sum: await a.sum().item(),
  };

  shutdown();
  return results;
}

function maxAbsDiff(x: Float32Array | number, y: Float32Array | number): number {
  if (typeof x === "number" && typeof y === "number") return Math.abs(x - y);
  if (typeof x === "number" || typeof y === "number") throw new Error("type mismatch");
  let m = 0;
  for (let i = 0; i < x.length; i++) {
    const d = Math.abs(x[i] - y[i]);
    if (d > m) m = d;
  }
  return m;
}

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
  transpose: 0,
  softmax: 1e-5,
  sum: 1e-2,
};

async function runParity(_: null, { log }: RunContext) {
  log("workers backend");
  const w = await runOps("workers");
  log("webgpu backend");
  const g = await runOps("webgpu");

  const rows: string[] = [];
  let allOk = true;
  for (const key of Object.keys(w) as (keyof Results)[]) {
    const diff = maxAbsDiff(w[key], g[key]);
    const tol = TOLERANCES[key];
    const ok = diff <= tol;
    if (!ok) allOk = false;
    rows.push(
      `  ${key.padEnd(10)} max|w-g| = ${diff.toExponential(3).padStart(11)}   tol ${tol.toExponential(1)}   ${ok ? "OK" : "FAIL"}`,
    );
  }
  for (const r of rows) log(r);

  return {
    pass: allOk,
    message: allOk ? "12/12 kernels match" : "some kernels diverged",
  };
}

defineTest({
  name: "WebGPU ↔ Workers kernel parity",
  params: [null],
  description: "Runs 12 kernels on both backends with fixed inputs and diffs their outputs.",
  runner: runParity,
});
