import { Tensor, init, shutdown } from "../src/index";

type Backend = "workers" | "wasm";

async function runOps(backend: Backend, threads: number) {
  await init({ backend, threadCount: threads, memorySizeMB: 64 });

  const aData = new Float32Array(64 * 64);
  const bData = new Float32Array(64 * 64);
  for (let i = 0; i < aData.length; i++) aData[i] = Math.sin(i * 0.13);
  for (let i = 0; i < bData.length; i++) bData[i] = Math.cos(i * 0.07);

  const a = Tensor.fromData(Array.from(aData), [64, 64]);
  const b = Tensor.fromData(Array.from(bData), [64, 64]);

  const results: Record<string, Float32Array | number> = {};
  results.add = await a.add(b).toArray();
  results.sub = await a.sub(b).toArray();
  results.mul = await a.mul(b).toArray();
  results.div = await a.div(b).toArray();
  results.neg = await a.neg().toArray();
  results.relu = await a.relu().toArray();
  results.exp = await a.exp().toArray();
  results.log = await a.mul(a).log().toArray(); // avoid log(negative)
  results.matmul = await a.matmul(b).toArray();
  results.transpose = await a.transpose().toArray();

  // Row-wise softmax
  results.softmax = await a.softmax(1).toArray();

  // SUM (two-phase reduce)
  results.sum = await a.sum().item();

  shutdown();
  return results;
}

function maxAbsDiff(x: Float32Array | number, y: Float32Array | number): number {
  if (typeof x === "number" && typeof y === "number") {
    return Math.abs(x - y);
  }
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

export async function runWasmSmoke(log: (msg: string) => void) {
  log("Running kernel suite on workers backend...");
  const workersResult = await runOps("workers", 4);
  log("Running kernel suite on wasm backend...");
  const wasmResult = await runOps("wasm", 4);

  const tolerances: Record<string, number> = {
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
    softmax: 1e-6,
    sum: 1e-2, // reduction ordering differs; float sum drift is expected
  };

  let allOk = true;
  for (const key of Object.keys(workersResult)) {
    const diff = maxAbsDiff(workersResult[key], wasmResult[key]);
    const tol = tolerances[key] ?? 1e-5;
    const ok = diff <= tol;
    if (!ok) allOk = false;
    log(`  ${key.padEnd(10)} max|w-a| = ${diff.toExponential(3).padStart(11)}   tol ${tol.toExponential(1)}   ${ok ? "OK" : "FAIL"}`);
  }
  log(allOk ? "ALL PASS" : "SOME FAILED");
  return allOk;
}
