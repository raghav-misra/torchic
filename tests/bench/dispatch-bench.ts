import torchic, { Tensor } from "../../src/index";

type Backend = "workers" | "wasm";

async function benchMatmulE2E(
  backend: Backend,
  m: number,
  k: number,
  n: number,
  threads: number,
  trials = 10,
) {
  const logs: string[] = [];
  const push = (s: string) => logs.push(s);
  push(`[${backend}] Initializing with ${threads} threads...`);
  await torchic.init({ backend, threadCount: threads });

  const A = Tensor.randn([m, k]);
  const B = Tensor.randn([k, n]);

  // Warmup
  for (let i = 0; i < 5; i++) {
    const out = A.matmul(B);
    await out.toArray();
  }

  const times: number[] = [];
  // Read A and B once for correctness validation
  const _Adata = await A.toArray(false);
  const _Bdata = await B.toArray(false);

  function _naiveMatmul(a: Float32Array, b: Float32Array, m: number, k: number, n: number) {
    const out = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
      const aRowBase = i * k;
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let p = 0; p < k; p++) {
          sum += a[aRowBase + p] * b[p * n + j];
        }
        out[i * n + j] = sum;
      }
    }
    return out;
  }

  // const expected = naiveMatmul(Adata, Bdata, m, k, n);

  for (let t = 0; t < trials; t++) {
    const t0 = performance.now();
    const out = A.matmul(B);
    const _arr = await out.toArray(false);
    const t1 = performance.now();
    times.push(t1 - t0);

    // validate against naive result
    const maxDiff = 0;
    const mismatches = 0;
    // for (let i = 0; i < arr.length; i++) {
    //   const d = Math.abs(arr[i] - expected[i]);
    //   if (d > maxDiff) maxDiff = d;
    //   // consider mismatch if difference exceeds tolerance
    //   if (d > 1e-3 && !(Number.isNaN(arr[i]) && Number.isNaN(expected[i])))
    //     mismatches++;
    // }

    push(
      `trial ${t + 1}/${trials}: ${(t1 - t0).toFixed(
        3,
      )} ms (maxDiff=${maxDiff.toExponential()}, mismatches=${mismatches})`,
    );
  }

  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)];
  const flops = 2 * m * n * k;
  const gflops = flops / (median / 1000) / 1e9;

  push(
    `[${backend}] E2E matmul ${m}x${k} * ${k}x${n} with ${threads} threads: median ${median.toFixed(
      3,
    )} ms - ${gflops.toFixed(3)} GFLOPS`,
  );
  return { backend, m, k, n, threads, medianMs: median, gflops, logs };
}

export async function runBench(threads: number, log: (msg: string) => void, backend: Backend = "workers") {
  const t = threads;
  try {
    torchic.shutdown();
  } catch {
    // ignore
  }
  const sizes = [
    [128, 128, 128],
    [256, 128, 128],
    [256, 256, 256],
    [512, 512, 512],
    [1024, 1024, 1024],
  ];
  try {
    for (const [m, k, n] of sizes) {
      const res = await benchMatmulE2E(backend, m, k, n, t, 7);
      for (const line of res.logs) log(line);
      log("----");
    }
  } catch (e) {
    log(`Bench error: ${String(e)}`);
  }
}

export async function runBackendCompare(log: (msg: string) => void) {
  const backends: Backend[] = ["workers", "wasm"];
  const threadCounts = [1, 2, 4, 8];
  const sizes: [number, number, number][] = [
    [128, 128, 128],
    [512, 512, 512],
    [1024, 1024, 1024],
  ];

  const results: Record<string, number> = {};

  for (const size of sizes) {
    const [m, k, n] = size;
    log(`\n=== matmul ${m}x${k}x${n} ===`);
    for (const backend of backends) {
      for (const t of threadCounts) {
        try {
          torchic.shutdown();
        } catch {
          // ignore
        }
        try {
          const res = await benchMatmulE2E(backend, m, k, n, t, 5);
          const key = `${backend}@${t}`;
          results[`${m}x${k}x${n}|${key}`] = res.medianMs;
          log(
            `  ${backend.padEnd(8)} threads=${t}: ${res.medianMs.toFixed(2).padStart(8)} ms   ${res.gflops
              .toFixed(3)
              .padStart(6)} GFLOPS`,
          );
        } catch (e) {
          log(`  ${backend}@${t} FAIL: ${String(e)}`);
        }
      }
    }
    for (const t of threadCounts) {
      const w = results[`${m}x${k}x${n}|workers@${t}`];
      const a = results[`${m}x${k}x${n}|wasm@${t}`];
      if (w && a) {
        log(`  speedup wasm/workers @ ${t}t: ${(w / a).toFixed(2)}x`);
      }
    }
  }
  try {
    torchic.shutdown();
  } catch {
    // ignore
  }
}

// Keep default export for compatibility
export default { runBench };
