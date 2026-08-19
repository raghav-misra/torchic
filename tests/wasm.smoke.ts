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

  const sum = a.add(b);
  const prod = a.matmul(b);

  const sumOut = await sum.toArray();
  const prodOut = await prod.toArray();

  shutdown();
  return { sumOut, prodOut };
}

function maxAbsDiff(x: Float32Array, y: Float32Array): number {
  let m = 0;
  for (let i = 0; i < x.length; i++) {
    const d = Math.abs(x[i] - y[i]);
    if (d > m) m = d;
  }
  return m;
}

export async function runWasmSmoke(log: (msg: string) => void) {
  log("Running ADD + MATMUL on workers backend...");
  const workersResult = await runOps("workers", 4);
  log(`  workers add[0..4]  = ${Array.from(workersResult.sumOut.slice(0, 4)).join(", ")}`);
  log(`  workers matmul[0..4] = ${Array.from(workersResult.prodOut.slice(0, 4)).join(", ")}`);

  log("Running ADD + MATMUL on wasm backend...");
  const wasmResult = await runOps("wasm", 4);
  log(`  wasm add[0..4]  = ${Array.from(wasmResult.sumOut.slice(0, 4)).join(", ")}`);
  log(`  wasm matmul[0..4] = ${Array.from(wasmResult.prodOut.slice(0, 4)).join(", ")}`);

  const addDiff = maxAbsDiff(workersResult.sumOut, wasmResult.sumOut);
  const mmDiff = maxAbsDiff(workersResult.prodOut, wasmResult.prodOut);

  log(`ADD    max |workers - wasm| = ${addDiff.toExponential(3)}`);
  log(`MATMUL max |workers - wasm| = ${mmDiff.toExponential(3)}`);

  const ok = addDiff < 1e-5 && mmDiff < 1e-3;
  log(ok ? "PASS" : "FAIL");
  return ok;
}
