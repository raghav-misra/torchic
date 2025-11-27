import { Tensor } from "../src/index";

async function run() {
  console.log("Initializing...");
  await Tensor.init(4);
  console.log("Initialized.");

  console.log("Creating tensors...");
  const a = Tensor.randn([100]);
  const b = Tensor.randn([100]);

  console.log("a", (await a.toArray()).slice(0, 10));
  console.log("b", (await b.toArray()).slice(0, 10));

  console.log("Computing c = a + b...");
  const c = a.add(b);

  console.log("Fetching result...");
  const result = await c.toArray();

    console.log("Result (first 10):", result.slice(0, 10));

    console.log("Testing MatMul...");
    const m1 = Tensor.randn([10, 20]);
    const m2 = Tensor.randn([20, 10]);
    const m3 = m1.matmul(m2);
    console.log("MatMul Result Shape:", m3.shape);
    const m3Data = await m3.toArray();
    console.log("MatMul Result (first 10):", m3Data.slice(0, 10));
    m3.setValue([0, 0], 0);

    // --- Verification ---
    console.log("Verifying MatMul result...");
    const m1Data = await m1.toArray();
    const m2Data = await m2.toArray();
    
    const M = 10, K = 20, N = 10;
    let maxError = 0;

    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            let sum = 0;
            for (let k = 0; k < K; k++) {
                sum += m1Data[i * K + k] * m2Data[k * N + j];
            }
            const libVal = m3Data[i * N + j];
            const diff = Math.abs(sum - libVal);
            if (diff > maxError) maxError = diff;
        }
    }

    console.log(`Max Error: ${maxError}`);
    if (maxError < 1e-4) {
        console.log("✅ MatMul Verification Passed!");
    } else {
        console.error("❌ MatMul Verification Failed!");
    }

    // --- Test getValue/setValue ---
    console.log("Testing getValue/setValue...");
    const t = Tensor.zeros([2, 2]);
    t.setValue([0, 1], 42);
    const val = await t.getValue([0, 1]);
    console.log(`Value at [0, 1]: ${val}`);
    if (val === 42) {
        console.log("✅ getValue/setValue Verification Passed!");
    } else {
        console.error(`❌ getValue/setValue Verification Failed! Expected 42, got ${val}`);
    }

    console.log("Done.");
}
run().catch(console.error);
