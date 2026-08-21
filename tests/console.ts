import torchic, { Tensor } from "../src/index";
import { benchOne, type Backend } from "./suites/matmul-bench";

interface TorchicBench {
  matmul(backend: Backend, size: number, threads?: number): Promise<{ medianMs: number; gflops: number }>;
  matmulShape(
    backend: Backend,
    m: number,
    k: number,
    n: number,
    threads?: number,
  ): Promise<{ medianMs: number; gflops: number }>;
  hot(
    backend: Backend,
    size: number,
    iterations?: number,
    threads?: number,
  ): Promise<number[]>;
}

declare global {
  interface Window {
    torchicBench: TorchicBench;
  }
}

// Init once, run many matmuls back-to-back so the GPU/CPU doesn't downclock
// between short bursts. Returns per-iteration GFLOPS so you can see the ramp.
async function hot(backend: Backend, size: number, iterations = 20, threads = 1): Promise<number[]> {
  const opts =
    backend === "webgpu"
      ? { backend, memorySizeMB: 512 }
      : { backend, threadCount: threads, memorySizeMB: 512 };
  await torchic.init(opts);
  const A = Tensor.randn([size, size]);
  const B = Tensor.randn([size, size]);
  const out: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    await A.matmul(B).toArray();
    const dt = performance.now() - t0;
    out.push((2 * size ** 3) / (dt / 1000) / 1e9);
  }
  torchic.shutdown();
  return out;
}

window.torchicBench = {
  matmul: (backend, size, threads = 1) => benchOne(backend, size, size, size, threads),
  matmulShape: (backend, m, k, n, threads = 1) => benchOne(backend, m, k, n, threads),
  hot,
};

console.log(
  "%ctorchicBench ready",
  "color: #63b3ed; font-weight: bold",
  "\n  await torchicBench.matmul('webgpu', 2048)",
  "\n  await torchicBench.matmul('wasm', 1024, 8)",
  "\n  await torchicBench.hot('webgpu', 2048, 50)   // sees the GPU boost ramp",
);
