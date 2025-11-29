import { Tensor } from "../../src";

async function benchMatmulE2E(
  m: number,
  k: number,
  n: number,
  threads: number,
  trials = 10
) {
  const logs: string[] = [];
  const push = (s: string) => logs.push(s);
  push(`Initializing Tensor runtime with ${threads} threads...`);
  await Tensor.init(threads);

  const A = Tensor.randn([m, k]);
  const B = Tensor.randn([k, n]);

  // Warmup
  for (let i = 0; i < 5; i++) {
    const out = A.matmul(B);
    await out.toArray();
  }

  const times: number[] = [];
  // Read A and B once for correctness validation
  const Adata = await A.toArray(false);
  const Bdata = await B.toArray(false);

  function naiveMatmul(
    a: Float32Array,
    b: Float32Array,
    m: number,
    k: number,
    n: number
  ) {
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

  const expected = naiveMatmul(Adata, Bdata, m, k, n);

  for (let t = 0; t < trials; t++) {
    const t0 = performance.now();
    const out = A.matmul(B);
    const arr = await out.toArray(false);
    const t1 = performance.now();
    times.push(t1 - t0);

    // validate against naive result
    let maxDiff = 0;
    let mismatches = 0;
    for (let i = 0; i < arr.length; i++) {
      const d = Math.abs(arr[i] - expected[i]);
      if (d > maxDiff) maxDiff = d;
      // consider mismatch if difference exceeds tolerance
      if (d > 1e-3 && !(Number.isNaN(arr[i]) && Number.isNaN(expected[i])))
        mismatches++;
    }

    push(
      `trial ${t + 1}/${trials}: ${(t1 - t0).toFixed(
        3
      )} ms (maxDiff=${maxDiff.toExponential()}, mismatches=${mismatches})`
    );
  }

  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)];
  const flops = 2 * m * n * k;
  const gflops = flops / (median / 1000) / 1e9;

  push(
    `E2E matmul ${m}x${k} * ${k}x${n} with ${threads} threads: median ${median.toFixed(
      3
    )} ms — ${gflops.toFixed(3)} GFLOPS`
  );
  return { m, k, n, threads, medianMs: median, gflops, logs };
}

export async function runBench(threads: number, log: (msg: string) => void) {
  const t = threads;
  // Ensure previous runtime (if any) is shut down so we can reinit with new thread count
  try {
    Tensor.shutdown();
  } catch (e) {
    // ignore
  }
  // sizes tuned to be realistic
  const sizes = [
    [128, 128, 128],
    [256, 128, 128],
    [256, 256, 256],
    // larger size for stress-testing cache / bandwidth scaling
    [512, 512, 512],
  ];
  try {
    for (const [m, k, n] of sizes) {
      const res = await benchMatmulE2E(m, k, n, t, 7);
      // forward collected logs to provided logger
      for (const line of res.logs) log(line);
      log("----");
    }
  } catch (e) {
    log(`Bench error: ${String(e)}`);
  }
}

// Keep default export for compatibility
export default { runBench };
