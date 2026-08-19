import torchic, { Tensor } from "../../src/index";
import { defineBench } from "../framework/define";
import type { BenchMetrics, RunContext } from "../framework/types";

type Backend = "workers" | "wasm";

const SIZES: [number, number, number][] = [
  [128, 128, 128],
  [512, 512, 512],
  [1024, 1024, 1024],
];

async function benchOne(backend: Backend, m: number, k: number, n: number, threads: number, trials = 5) {
  await torchic.init({ backend, threadCount: threads });
  const A = Tensor.randn([m, k]);
  const B = Tensor.randn([k, n]);
  for (let i = 0; i < 3; i++) await A.matmul(B).toArray();

  const times: number[] = [];
  for (let t = 0; t < trials; t++) {
    const t0 = performance.now();
    await A.matmul(B).toArray();
    times.push(performance.now() - t0);
  }
  torchic.shutdown();

  times.sort((a, b) => a - b);
  const medianMs = times[Math.floor(times.length / 2)];
  const gflops = (2 * m * n * k) / (medianMs / 1000) / 1e9;
  return { medianMs, gflops };
}

async function runMatmul(threads: number, { log }: RunContext): Promise<BenchMetrics> {
  const metrics: BenchMetrics = {};
  for (const [m, k, n] of SIZES) {
    const label = `${m}³`;
    log(`workers ${label} threads=${threads}...`);
    const w = await benchOne("workers", m, k, n, threads);
    log(`  ${w.medianMs.toFixed(2)} ms   ${w.gflops.toFixed(2)} GFLOPS`);

    log(`wasm ${label} threads=${threads}...`);
    const a = await benchOne("wasm", m, k, n, threads);
    log(`  ${a.medianMs.toFixed(2)} ms   ${a.gflops.toFixed(2)} GFLOPS`);

    metrics[`workers ${label}`] = w.gflops.toFixed(2);
    metrics[`wasm ${label}`] = a.gflops.toFixed(2);
    metrics[`speedup ${label}`] = (a.gflops / w.gflops).toFixed(2) + "×";
  }
  return metrics;
}

defineBench({
  name: "Matmul – Workers vs WASM (GFLOPS)",
  paramName: "threads",
  params: [1, 2, 4, 8],
  description: "E2E matmul through the dispatcher on both backends across three sizes.",
  highlight: SIZES.map(([m]) => `speedup ${m}³`),
  runner: runMatmul,
});
