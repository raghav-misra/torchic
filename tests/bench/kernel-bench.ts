import { matmul as matmulKernel } from "../../src/backend/workers/kernels/matmul";
import { add as addKernel } from "../../src/backend/workers/kernels/elementwise";

function randArray(len: number) {
  const a = new Float32Array(len);
  for (let i = 0; i < len; i++) a[i] = Math.random() * 2 - 1;
  return a;
}

function benchMatmul(m: number, k: number, n: number, trials = 10) {
  const A = randArray(m * k);
  const B = randArray(k * n);
  const out = new Float32Array(m * n);

  // warmup
  for (let i = 0; i < 5; i++) matmulKernel(A, B, out, m, n, k, 0, m);

  const times: number[] = [];
  for (let t = 0; t < trials; t++) {
    const t0 = performance.now();
    matmulKernel(A, B, out, m, n, k, 0, m);
    const t1 = performance.now();
    times.push(t1 - t0);
  }

  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)];
  const flops = 2 * m * n * k;
  const gflops = flops / (median / 1000) / 1e9;
  return { m, k, n, medianMs: median, gflops, times };
}

function benchAdd(n: number, trials = 20) {
  const A = randArray(n);
  const B = randArray(n);
  const out = new Float32Array(n);

  for (let i = 0; i < 5; i++) addKernel(A, B, out, 0, n);

  const times: number[] = [];
  for (let t = 0; t < trials; t++) {
    const t0 = performance.now();
    addKernel(A, B, out, 0, n);
    const t1 = performance.now();
    times.push(t1 - t0);
  }

  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)];
  return { n, medianMs: median, opsPerSec: n / (median / 1000), times };
}

async function main() {
  console.log("Kernel microbench — matmul and elementwise (Node)");

  const matSizes = [64, 128, 256];
  for (const s of matSizes) {
    const res = benchMatmul(s, s, s, 7);
    console.log(
      `matmul ${s}x${s}: median ${res.medianMs.toFixed(
        3
      )} ms — ${res.gflops.toFixed(3)} GFLOPS`
    );
  }

  const elems = [1e5, 1e6, 5e6].map((v) => Math.floor(v));
  for (const n of elems) {
    const res = benchAdd(n, 10);
    console.log(
      `add ${n}: median ${res.medianMs.toFixed(3)} ms — ${Math.round(
        res.opsPerSec
      ).toLocaleString()} ops/sec`
    );
  }
}

main().catch((e) => {
  console.error(e);
});
