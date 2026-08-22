import torchic, { Tensor } from "../../src/index";
import { defineBench } from "../framework/define";
import type { BenchMetrics, RunContext } from "../framework/types";

type Backend = "workers" | "wasm" | "webgpu";

const SIZES: [number, number, number][] = [
  [128, 128, 128],
  [512, 512, 512],
  [1024, 1024, 1024],
  [2048, 2048, 2048],
];

async function benchOne(
  backend: Backend,
  m: number,
  k: number,
  n: number,
  threads: number,
) {
  // 2048³ takes seconds per matmul on workers @ low thread counts, so scale
  // warmup + trials down for very large workloads.
  const isLarge = m * n * k >= 1024 * 1024 * 1024;
  const warmup = isLarge ? 1 : 3;
  const trials = isLarge ? 3 : 5;

  const opts =
    backend === "webgpu"
      ? { backend, memorySizeMB: 512 }
      : { backend, threadCount: threads, memorySizeMB: 512 };
  await torchic.init(opts);
  const A = Tensor.randn([m, k]);
  const B = Tensor.randn([k, n]);
  for (let i = 0; i < warmup; i++) await A.matmul(B).toArray();

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

export { benchOne };
export type { Backend };

const webgpuCache = new Map<string, { medianMs: number; gflops: number }>();

// Right-align a number to a fixed column width in monospace.
const fmt = (v: number, w = 8, digits = 2) => v.toFixed(digits).padStart(w);

async function runMatmul(threads: number, { log }: RunContext): Promise<BenchMetrics> {
  const metrics: BenchMetrics = {};
  for (const [m, k, n] of SIZES) {
    const sizeLabel = `${m}³`;

    log(`workers ${sizeLabel} threads=${threads}...`);
    const w = await benchOne("workers", m, k, n, threads);
    log(`  ${w.medianMs.toFixed(2)} ms   ${w.gflops.toFixed(2)} GFLOPS`);

    log(`wasm ${sizeLabel} threads=${threads}...`);
    const a = await benchOne("wasm", m, k, n, threads);
    log(`  ${a.medianMs.toFixed(2)} ms   ${a.gflops.toFixed(2)} GFLOPS`);

    let g = webgpuCache.get(sizeLabel);
    if (!g) {
      log(`webgpu ${sizeLabel} (single dispatch)...`);
      g = await benchOne("webgpu", m, k, n, 1);
      webgpuCache.set(sizeLabel, g);
      log(`  ${g.medianMs.toFixed(2)} ms   ${g.gflops.toFixed(2)} GFLOPS`);
    }

    metrics[`${sizeLabel} GFLOPS`] =
      `Workers ${fmt(w.gflops)}\n` +
      `WASM    ${fmt(a.gflops)}\n` +
      `WebGPU  ${fmt(g.gflops)}`;

    metrics[`${sizeLabel} speedups`] =
      `wasm/workers ${fmt(a.gflops / w.gflops, 6)}×\n` +
      `gpu/wasm     ${fmt(g.gflops / a.gflops, 6)}×`;
  }
  return metrics;
}

defineBench({
  name: "Matmul: Workers vs WASM vs WebGPU (GFLOPS)",
  paramName: "threads",
  params: [1, 2, 4, 8],
  description:
    "E2E matmul on all three backends. WebGPU value is cached per size; thread count doesn't affect it.",
  highlight: SIZES.map(([m]) => `${m}³ speedups`),
  runner: runMatmul,
});
