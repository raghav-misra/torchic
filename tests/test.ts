import { Tensor, noGrad } from "../src/index";
import { addTest, addInfo } from "./testUtils";

async function runTests() {
  await addTest("Initialization", async () => {
    await Tensor.init(4);
    return [true, "Initialized with 4 threads"];
  });

  await addTest("Basic Arithmetic (Add)", async (log) => {
    log("Testing element-wise addition");
    const a = Tensor.fromData([1, 2, 3], [3]);
    const b = Tensor.fromData([4, 5, 6], [3]);
    log(`a = [${await a.toArray()}]`);
    log(`b = [${await b.toArray()}]`);

    const c = a.add(b);
    const res = await c.toArray();
    log(`c = a + b = [${res}]`);

    const expected = [5, 7, 9];
    const match = res.every((val, i) => Math.abs(val - expected[i]) < 1e-5);

    return [
      match,
      match ? `Got [${res}]` : `Expected [${expected}], got [${res}]`,
    ];
  });

  await addTest("Matrix Multiplication", async (log) => {
    const M = 10,
      K = 20,
      N = 10;
    log(`Testing ${M}x${K} * ${K}x${N} matrix multiplication`);
    const m1 = Tensor.randn([M, K]);
    const m2 = Tensor.randn([K, N]);
    const m3 = m1.matmul(m2);
    log(`Result shape: ${m3.shape}`);

    log("Verifying result against manual computation...");
    const m3Data = await m3.toArray();
    const m1Data = await m1.toArray();
    const m2Data = await m2.toArray();

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

    return [maxError < 1e-4, `Max Error: ${maxError}`];
  });

  await addTest("Get/Set Value", async (log) => {
    log("Creating 2x2 zero tensor");
    const t = Tensor.zeros([2, 2]);
    log("Setting value at [0, 1] to 42");
    t.setValue([0, 1], 42);
    const val = await t.getValue([0, 1]);
    log(`Retrieved value: ${val}`);

    return [val === 42, `Expected 42, got ${val}`];
  });

  await addTest("Autograd (y = x^2 + 3x)", async (log) => {
    // y = x^2 + 3x
    // dy/dx = 2x + 3
    // Let x = 2. Then y = 4 + 6 = 10. dy/dx = 7.

    log("Computing y = x^2 + 3x where x = 2");
    const x = Tensor.fromData([2], [1], true);
    const x2 = x.mul(x);
    const threeX = x.mul(Tensor.fromData([3], [1]));
    const y = x2.add(threeX);

    const yVal = await y.item();
    log(`Forward: y = ${yVal} (Expected 10)`);

    log("Running backward pass...");
    y.backward();

    if (!x.grad) return [false, "x.grad is null"];

    const gradVal = await x.grad.item();
    log(`Gradient: dy/dx = ${gradVal} (Expected 7)`);
    const err = Math.abs(gradVal - 7);

    return [err < 1e-4, `y=${yVal}, dy/dx=${gradVal} (Expected 7)`];
  });

  await addTest("MatMul Gradient (Dot Product)", async (log) => {
    // A (1x2) * B (2x1) -> C (1x1)
    log("Testing matrix multiplication gradients");
    const a = Tensor.fromData([1, 2], [1, 2], true);
    const b = Tensor.fromData([3, 4], [2, 1], true);
    log(`A = [${await a.toArray()}] (shape: ${a.shape})`);
    log(`B = [${await b.toArray()}] (shape: ${b.shape})`);

    const c = a.matmul(b);
    const cVal = await c.item();
    log(`C = A * B = ${cVal} (Expected: 11)`);

    if (Math.abs(cVal - 11) > 1e-4)
      return [false, `Forward failed. Expected 11, got ${cVal}`];

    log("Computing gradients...");
    c.backward();

    if (!a.grad || !b.grad) return [false, "Gradients missing"];

    const aGrad = await a.grad.toArray(); // Expected B^T = [3, 4]
    const bGrad = await b.grad.toArray(); // Expected A^T = [1, 2]
    log(`dA = [${aGrad}] (Expected: [3, 4])`);
    log(`dB = [${bGrad}] (Expected: [1, 2])`);

    const aOk = Math.abs(aGrad[0] - 3) < 1e-4 && Math.abs(aGrad[1] - 4) < 1e-4;
    const bOk = Math.abs(bGrad[0] - 1) < 1e-4 && Math.abs(bGrad[1] - 2) < 1e-4;

    if (!aOk) return [false, `dA failed. Expected [3, 4], got [${aGrad}]`];
    if (!bOk) return [false, `dB failed. Expected [1, 2], got [${bGrad}]`];

    return [true, "Forward and Backward correct"];
  });

  await addTest("MLP Layer (ReLU + MatMul + Bias)", async (log) => {
    // x: [1, 2]
    // W: [[0.1, 0.2], [-0.1, 0.3]]
    // b: [0.1, -0.1]
    // z = xW + b
    //   = [1*0.1 + 2*-0.1, 1*0.2 + 2*0.3] + [0.1, -0.1]
    //   = [-0.1, 0.8] + [0.1, -0.1]
    //   = [0.0, 0.7]
    // a = relu(z) = [0.0, 0.7]
    // Loss = sum(a) = a * [1, 1]^T = 0.7

    log("Testing MLP forward and backward pass");
    const x = Tensor.fromData([1, 2], [1, 2], true);
    const W = Tensor.fromData([0.1, 0.2, -0.1, 0.3], [2, 2], true);
    const b = Tensor.fromData([0.1, -0.1], [1, 2], true);

    log("Forward: z = xW + b");
    const z = x.matmul(W).add(b);
    log(`z = [${await z.toArray()}]`);

    log("Activation: a = relu(z)");
    const a = z.relu();
    log(`a = [${await a.toArray()}]`);

    // Mock sum by matmul with ones
    const ones = Tensor.fromData([1, 1], [2, 1]);
    const loss = a.matmul(ones); // Scalar (1x1)

    const lossVal = await loss.item();
    log(`Loss = ${lossVal} (Expected: 0.7)`);
    if (Math.abs(lossVal - 0.7) > 1e-4)
      return [false, `Forward failed. Expected 0.7, got ${lossVal}`];

    log("Running backward pass...");
    loss.backward();

    // Gradients
    // dL/da = [1, 1]
    // dL/dz = dL/da * relu'(z) = [1, 1] * (z>0 ? 1 : 0). z=[0.0, 0.7].
    // dL/dz = [0, 1]

    // dL/db = dL/dz = [0, 1]
    // dL/dW = x^T * dL/dz = [1, 2]^T * [0, 1] = [[0, 1], [0, 2]]
    // dL/dx = dL/dz * W^T = [0, 1] * [[0.1, -0.1], [0.2, 0.3]] = [0.2, 0.3]

    if (!W.grad || !b.grad || !x.grad) return [false, "Gradients missing"];

    const wGrad = await W.grad.toArray();
    const bGrad = await b.grad.toArray();
    const xGrad = await x.grad.toArray();

    log(`dW = [${wGrad}] (Expected: [0, 1, 0, 2])`);
    log(`db = [${bGrad}] (Expected: [0, 1])`);
    log(`dx = [${xGrad}] (Expected: [0.2, 0.3])`);

    const wExpected = [0, 1, 0, 2];
    const bExpected = [0, 1];
    const xExpected = [0.2, 0.3];

    const check = (arr: Float32Array, exp: number[]) =>
      arr.every((v, i) => Math.abs(v - exp[i]) < 1e-4);

    if (!check(wGrad, wExpected))
      return [false, `dW failed. Expected ${wExpected}, got ${wGrad}`];
    if (!check(bGrad, bExpected))
      return [false, `db failed. Expected ${bExpected}, got ${bGrad}`];
    if (!check(xGrad, xExpected))
      return [false, `dx failed. Expected ${xExpected}, got ${xGrad}`];

    return [true, "MLP Layer Gradients correct"];
  });

  await addTest("Reductions (Sum/Mean)", async (log) => {
    log("Testing sum and mean reductions");
    const a = Tensor.fromData([1, 2, 3, 4], [4], true);
    log(`Input: [${await a.toArray()}]`);

    const s = a.sum();
    const m = a.mean();

    const sVal = await s.item();
    const mVal = await m.item();
    log(`Sum: ${sVal} (Expected: 10)`);
    log(`Mean: ${mVal} (Expected: 2.5)`);

    if (Math.abs(sVal - 10) > 1e-4)
      return [false, `Sum failed. Expected 10, got ${sVal}`];
    if (Math.abs(mVal - 2.5) > 1e-4)
      return [false, `Mean failed. Expected 2.5, got ${mVal}`];

    // Backward
    // d(mean)/da = 1/N = 0.25
    log("Computing gradient of mean...");
    m.backward();

    if (!a.grad) return [false, "Grad missing"];
    const grad = await a.grad.toArray();
    log(`Gradient: [${grad}] (Expected: all 0.25)`);
    const expected = [0.25, 0.25, 0.25, 0.25];

    const match = grad.every((v) => Math.abs(v - 0.25) < 1e-4);
    return [
      match,
      match ? "Passed" : `Grad failed. Expected ${expected}, got ${grad}`,
    ];
  });

  await addTest("Broadcasting Gradient (Add)", async (log) => {
    // x: [2, 3]
    // b: [3]
    // y = x + b
    // loss = y.sum()

    log("Testing broadcasting with gradient computation");
    const x = Tensor.fromData([1, 1, 1, 1, 1, 1], [2, 3], true);
    const b = Tensor.fromData([2, 2, 2], [3], true);
    log(`x shape: ${x.shape}, b shape: ${b.shape}`);

    const y = x.add(b);
    log(`y = x + b, shape: ${y.shape}`);
    const loss = y.sum();
    log(`loss = sum(y) = ${await loss.item()}`);

    log("Computing gradients...");
    loss.backward();

    if (!x.grad || !b.grad) return [false, "Gradients missing"];

    const xGrad = await x.grad.toArray();
    const bGrad = await b.grad.toArray();
    log(`x.grad: [${xGrad}] (Expected: all 1s)`);
    log(`b.grad: [${bGrad}] (Expected: all 2s, sum over broadcast dim)`);

    // x.grad should be all 1s (shape [2, 3])
    const xOk = xGrad.every((v) => Math.abs(v - 1.0) < 1e-4);

    // b.grad should be all 2s (shape [3]) because it was broadcasted across dim 0 (size 2)
    const bOk = bGrad.every((v) => Math.abs(v - 2.0) < 1e-4);

    if (!xOk) return [false, `x.grad failed. Expected all 1s, got ${xGrad}`];
    if (!bOk) return [false, `b.grad failed. Expected all 2s, got ${bGrad}`];

    return [true, "Broadcasting gradients correct"];
  });

  await addTest("noGrad Mode", async (log) => {
    log("Testing noGrad context manager");
    const x = Tensor.fromData([2], [1], true);

    let y: Tensor;
    log("Computing y = x^2 inside noGrad()...");
    await noGrad(async () => {
      y = x.mul(x);
    });

    log(`y.requiresGrad: ${y!.requiresGrad} (Expected: false)`);
    log(`y.op: ${y!.op} (Expected: null)`);

    // y should not require grad
    if (y!.requiresGrad) return [false, "y.requiresGrad should be false"];
    if (y!.op !== null) return [false, "y.op should be null"];

    // Backward should do nothing (or at least not crash, but since requiresGrad is false it returns early)
    log("Calling backward (should be no-op)...");
    y!.backward();

    if (x.grad) return [false, "x.grad should be null"];
    log("✓ x.grad is null as expected");

    return [true, "noGrad works"];
  });

  await addTest("Reshape (Zero-Copy)", async (log) => {
    log("Testing reshape zero-copy...");
    const a = Tensor.fromData([1, 2, 3, 4, 5, 6], [2, 3], true);
    log(`Original shape: ${a.shape}`);
    log(`Original ID: ${a.id}`);

    // Verify reshape creates a view (different ID, same memory)
    const b = a.reshape([3, 2]);
    log(`Reshaped shape: ${b.shape}`);
    log(`Reshaped ID: ${b.id}`);

    if (a.id === b.id) {
      log("❌ Reshape should create view with new ID");
      return [false, `Reshape should create view with new ID`];
    }
    log(`✓ Different IDs (view created)`);

    const bData = await b.toArray();
    log(`Reshaped data: [${bData}]`);
    log(`Expected: [1, 2, 3, 4, 5, 6]`);

    // Should be same data, different shape
    const match = bData.every((v, i) => v === i + 1);
    if (!match) return [false, "Reshape data mismatch"];

    // Backward
    const loss = b.sum();
    loss.backward();

    if (!a.grad) return [false, "Grad missing"];
    const grad = await a.grad.toArray();

    // Grad of sum is 1s. Reshape back to [2, 3] is still 1s.
    const gradMatch = grad.every((v) => v === 1);

    return [gradMatch, "Reshape forward/backward correct (zero-copy view)"];
  });

  await addTest("Transpose (Zero-Copy)", async (log) => {
    log("Testing transpose zero-copy...");
    const a = Tensor.fromData([1, 2, 3, 4, 5, 6], [2, 3]);
    log(`Original shape: ${a.shape}`);
    log(`Original ID: ${a.id}`);

    const b = a.transpose();
    log(`Transposed shape: ${b.shape}`);
    log(`Transposed ID: ${b.id}`);

    // Verify transpose creates a view (different ID, shared memory)
    if (a.id === b.id) {
      log("❌ Transpose should create view with new ID");
      return [false, `Transpose should create view with new ID`];
    }
    log(`✓ Different IDs (view created)`);

    const bData = await b.toArray();
    // Expected: [[1,2,3], [4,5,6]]^T = [[1,4], [2,5], [3,6]]
    // Flat: [1,4,2,5,3,6]
    const expected = [1, 4, 2, 5, 3, 6];
    log(`Transposed data: [${bData}]`);
    log(`Expected: [${expected}]`);

    const match = bData.every((val, i) => Math.abs(val - expected[i]) < 1e-5);
    return [
      match,
      match
        ? "Transpose correct (zero-copy view)"
        : `Expected ${expected}, got ${bData}`,
    ];
  });

  await addTest("Broadcasting (Add)", async (log) => {
    // A: [2, 3]
    // B: [3] -> Broadcast to [2, 3]
    log("Testing broadcasting during addition");
    const a = Tensor.fromData([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = Tensor.fromData([10, 20, 30], [3]);
    log(`A shape: ${a.shape}, B shape: ${b.shape}`);

    const c = a.add(b);
    const res = await c.toArray();
    log(`Result: [${res}]`);

    // Expected:
    // [1+10, 2+20, 3+30, 4+10, 5+20, 6+30]
    // [11, 22, 33, 14, 25, 36]
    const expected = [11, 22, 33, 14, 25, 36];
    log(`Expected: [${expected}]`);

    const match = res.every((val, i) => Math.abs(val - expected[i]) < 1e-5);
    return [match, match ? "Passed" : `Expected ${expected}, got ${res}`];
  });

  await addTest("Softmax Layer", async (log) => {
    // Input: [1, 4]
    log("Testing softmax activation");
    const xData = [0.1, 0.2, 0.3, 0.4];
    const x = Tensor.fromData(xData, [1, 4]);
    log(`Input: [${xData}]`);

    // Weights: [4, 3]
    // Let's use fixed values to verify math
    const wData = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3];
    const W = Tensor.fromData(wData, [4, 3]);

    // Bias: [1, 3]
    const bData = [0.1, 0.2, 0.3];
    const b = Tensor.fromData(bData, [1, 3]);

    // Linear Layer
    log("Computing z = xW + b");
    const z = x.matmul(W).add(b);

    // Manual Verification of Linear Layer
    // z[0] = (0.1*0.1 + 0.2*0.4 + 0.3*0.7 + 0.4*0.1) + 0.1 = (0.01 + 0.08 + 0.21 + 0.04) + 0.1 = 0.34 + 0.1 = 0.44
    // z[1] = (0.1*0.2 + 0.2*0.5 + 0.3*0.8 + 0.4*0.2) + 0.2 = (0.02 + 0.10 + 0.24 + 0.08) + 0.2 = 0.44 + 0.2 = 0.64
    // z[2] = (0.1*0.3 + 0.2*0.6 + 0.3*0.9 + 0.4*0.3) + 0.3 = (0.03 + 0.12 + 0.27 + 0.12) + 0.3 = 0.54 + 0.3 = 0.84

    const zData = await z.toArray();
    log(`z = [${Array.from(zData).map((v) => v.toFixed(2))}]`);
    const expectedZ = [0.44, 0.64, 0.84];
    log(`Expected z: [${expectedZ}]`);
    const zMatch = zData.every((v, i) => Math.abs(v - expectedZ[i]) < 1e-5);

    if (!zMatch)
      return [
        false,
        `Linear layer failed. Expected ${expectedZ}, got ${zData}`,
      ];

    // Softmax
    log("Applying softmax...");
    const probs = z.softmax();

    const probsData = await probs.toArray();
    log(`Probabilities: [${Array.from(probsData).map((p) => p.toFixed(4))}]`);
    const sumProbs = probsData.reduce((a, b) => a + b, 0);
    log(`Sum of probabilities: ${sumProbs.toFixed(6)} (Expected: 1.0)`);

    // Check if sum is close to 1
    const isSumOne = Math.abs(sumProbs - 1.0) < 1e-4;

    // Check if all probs are positive
    const arePositive = probsData.every((p) => p >= 0);

    return [isSumOne && arePositive, `Sum: ${sumProbs}, Probs: [${probsData}]`];
  });

  await addTest("Linear Regression (y = 2x + 1)", async (log) => {
    log("Target Function: y = 2x + 1");
    log("Generating synthetic data (N=20)...");

    // Generate Data
    const N = 20;
    const X_data = new Float32Array(N);
    const Y_data = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const val = Math.random() * 10;
      X_data[i] = val;
      Y_data[i] = 2 * val + 1 + (Math.random() - 0.5) * 0.1; // Add small noise
    }

    const x = Tensor.fromData(X_data, [N, 1]);
    const y_true = Tensor.fromData(Y_data, [N, 1]);

    log(`X-values: ${JSON.stringify(await x.toArray(), null, 2)}`);

    log(`Y-values: ${JSON.stringify(await y_true.toArray(), null, 2)}`);

    // Initialize Parameters
    const w = Tensor.randn([1, 1], true);
    const b = Tensor.zeros([1], true);

    log(
      `Initial w: ${(await w.item()).toFixed(4)}, b: ${(await b.item()).toFixed(
        4
      )}`
    );

    let learningRate = 0.01; // Start with higher LR
    const epochs = 2000;
    const lossThreshold = 0.005; // Stop if loss gets this low
    const gradClip = 1.0; // Clip gradients to prevent explosions

    log("\nTraining...");

    let finalEpoch = epochs;
    let converged = false;
    let prevLoss = Infinity;
    let prevW: Float32Array | null = null;
    let prevB: number | null = null;

    for (let i = 1; i <= epochs; i++) {
      // Forward
      const y_pred = x.matmul(w).add(b);

      // Loss (MSE)
      const diff = y_pred.sub(y_true);
      const loss = diff.mul(diff).mean();

      const lossVal = await loss.item();

      if (i % 100 === 0 || i === 1) {
        log(`Epoch ${i}: Loss = ${lossVal.toFixed(6)}, LR = ${learningRate.toFixed(6)}`);
      }

      // Check for NaN/Infinity
      if (isNaN(lossVal) || !isFinite(lossVal)) {
        log(`❌ Training diverged at epoch ${i} (Loss = ${lossVal})`);
        finalEpoch = i;
        converged = false;
        break;
      }

      // Detect explosion: loss increased by more than 50%
      if (i > 1 && lossVal > prevLoss * 1.5) {
        log(`⚠️  Loss exploded at epoch ${i} (${prevLoss.toFixed(6)} → ${lossVal.toFixed(6)})`);
        
        // Backtrack: restore previous weights
        if (prevW !== null && prevB !== null) {
          await noGrad(async () => {
            const wData = await w.toArray();
            for (let j = 0; j < wData.length; j++) {
              w.setValue([j], prevW![j]);
            }
            b.setValue([0], prevB!);
          });
          log(`   Restored previous weights`);
        }
        
        // Halve the learning rate
        learningRate /= 2;
        log(`   Reduced LR: ${(learningRate * 2).toFixed(6)} → ${learningRate.toFixed(6)}`);
        
        // Skip this iteration
        continue;
      }

      if (lossVal < lossThreshold) {
        log(
          `✅ Converged at epoch ${i} (Loss = ${lossVal.toFixed(
            6
          )} < ${lossThreshold})`
        );
        finalEpoch = i;
        converged = true;
        break;
      }

      // Save current state before updating
      prevW = await w.toArray();
      prevB = await b.item();
      prevLoss = lossVal;

      // Backward
      loss.backward();

      // Update with gradient clipping
      const lr = Tensor.fromData([learningRate], [1]);
      await noGrad(async () => {
        // Clip w gradient
        if (w.grad) {
          const wGradData = await w.grad.toArray();
          const wGradNorm = Math.sqrt(
            wGradData.reduce((sum, v) => sum + v * v, 0)
          );
          if (wGradNorm > gradClip) {
            w.grad = w.grad.mul(Tensor.fromData([gradClip / wGradNorm], [1]));
          }
          w.sub_(w.grad.mul(lr));
        }

        // Clip b gradient
        if (b.grad) {
          const bGradData = await b.grad.toArray();
          const bGradNorm = Math.sqrt(
            bGradData.reduce((sum, v) => sum + v * v, 0)
          );
          if (bGradNorm > gradClip) {
            b.grad = b.grad.mul(Tensor.fromData([gradClip / bGradNorm], [1]));
          }
          b.sub_(b.grad.mul(lr));
        }

        // Zero grads
        w.grad = null;
        b.grad = null;
      });
    }

    const finalW = await w.item();
    const finalB = await b.item();

    log(`\nFinal w: ${finalW.toFixed(4)} (Expected ~2.0)`);
    log(`Final b: ${finalB.toFixed(4)} (Expected ~1.0)`);
    log(`Stopped at epoch ${finalEpoch}`);

    const success =
      converged &&
      !isNaN(finalW) &&
      !isNaN(finalB) &&
      Math.abs(finalW - 2) < 0.15 &&
      Math.abs(finalB - 1) < 0.4;
    if (success) {
      log("✅ Converged successfully!");
    } else {
      log("❌ Failed to converge.");
    }

    return [success, success ? "Converged" : "Failed to converge"];
  });

  await addTest("CrossEntropy Loss", async (log) => {
    // Logits: [1.0, 2.0, 3.0]
    // Target: [0, 0, 1]
    log("Testing cross-entropy loss computation");
    const logits = Tensor.fromData([1.0, 2.0, 3.0], [1, 3], true);
    const target = Tensor.fromData([0, 0, 1], [1, 3]);
    log(`Logits: [${await logits.toArray()}]`);
    log(`Target: [${await target.toArray()}] (one-hot)`);

    const loss = Tensor.crossEntropy(logits, target);
    const lossVal = await loss.item();
    log(`Loss: ${lossVal.toFixed(4)}`);

    // Expected:
    // exp: [2.718, 7.389, 20.085]
    // sum: 30.192
    // p2: 20.085 / 30.192 = 0.6652
    // -log(0.6652) = 0.4076
    const expected = 0.4076;
    log(`Expected: ${expected}`);

    const match = Math.abs(lossVal - expected) < 1e-3;

    // Backward
    log("Computing gradients...");
    loss.backward();
    if (!logits.grad) return [false, "Grad missing"];

    // dLoss/dLogits = probs - target
    // p0 - 0 = 0.0900
    // p1 - 0 = 0.2447
    // p2 - 1 = 0.6652 - 1 = -0.3348

    const grad = await logits.grad.toArray();
    log(`Gradient: [${Array.from(grad).map((g) => g.toFixed(4))}]`);
    const gExpected = [0.09, 0.2447, -0.3348];
    log(`Expected: [${gExpected}]`);

    const gradMatch = grad.every((v, i) => Math.abs(v - gExpected[i]) < 1e-3);

    return [match && gradMatch, `Loss: ${lossVal.toFixed(4)}, Grad: [${grad}]`];
  });

  await addTest("Multivariate Linear Regression (N=50, D=3)", async (log) => {
    log("Target: y = 1.5x1 - 2.0x2 + 1.0x3 + 0.5");

    const N = 50;
    const D = 3;

    // True parameters: W=[1.5, -2.0, 1.0], b=0.5
    const W_true = [1.5, -2.0, 1.0];
    const b_true = 0.5;

    // Generate Data
    const X_data = new Float32Array(N * D);
    const Y_data = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      let sum = 0;
      for (let j = 0; j < D; j++) {
        const val = Math.random() * 2; // Range [0, 2]
        X_data[i * D + j] = val;
        sum += val * W_true[j];
      }
      Y_data[i] = sum + b_true + (Math.random() - 0.5) * 0.1;
    }

    const x = Tensor.fromData(X_data, [N, D]);
    const y_true = Tensor.fromData(Y_data, [N, 1]);

    // Initialize Parameters
    const w = Tensor.randn([D, 1], true);
    const b = Tensor.zeros([1], true);

    let learningRate = 0.01; // Start with higher LR
    const epochs = 3000; // More epochs
    const lossThreshold = 0.008; // Stop if loss gets this low
    const gradClip = 1.0; // Clip gradients to prevent explosions

    log("Training...");

    let finalEpoch = epochs;
    let converged = false;
    let prevLoss = Infinity;
    let prevW: Float32Array | null = null;
    let prevB: number | null = null;

    for (let i = 1; i <= epochs; i++) {
      const y_pred = x.matmul(w).add(b);
      const diff = y_pred.sub(y_true);
      const loss = diff.mul(diff).mean();

      const lossVal = await loss.item();

      if (i % 150 === 0 || i === 1) {
        log(`Epoch ${i}: Loss = ${lossVal.toFixed(6)}, LR = ${learningRate.toFixed(6)}`);
      }

      // Check for NaN/Infinity
      if (isNaN(lossVal) || !isFinite(lossVal)) {
        log(`❌ Training diverged at epoch ${i} (Loss = ${lossVal})`);
        finalEpoch = i;
        converged = false;
        break;
      }

      // Detect explosion: loss increased by more than 50%
      if (i > 1 && lossVal > prevLoss * 1.5) {
        log(`⚠️  Loss exploded at epoch ${i} (${prevLoss.toFixed(6)} → ${lossVal.toFixed(6)})`);
        
        // Backtrack: restore previous weights
        if (prevW !== null && prevB !== null) {
          await noGrad(async () => {
            const wData = await w.toArray();
            for (let j = 0; j < wData.length; j++) {
              w.setValue([j], prevW![j]);
            }
            b.setValue([0], prevB!);
          });
          log(`   Restored previous weights`);
        }
        
        // Halve the learning rate
        learningRate /= 2;
        log(`   Reduced LR: ${(learningRate * 2).toFixed(6)} → ${learningRate.toFixed(6)}`);
        
        // Skip this iteration
        continue;
      }

      if (lossVal < lossThreshold) {
        log(
          `✅ Converged at epoch ${i} (Loss = ${lossVal.toFixed(
            6
          )} < ${lossThreshold})`
        );
        finalEpoch = i;
        converged = true;
        break;
      }

      // Save current state before updating
      prevW = await w.toArray();
      prevB = await b.item();
      prevLoss = lossVal;

      loss.backward();

      // Update with gradient clipping
      const lr = Tensor.fromData([learningRate], [1]);
      await noGrad(async () => {
        // Clip w gradient
        if (w.grad) {
          const wGradData = await w.grad.toArray();
          const wGradNorm = Math.sqrt(
            wGradData.reduce((sum, v) => sum + v * v, 0)
          );
          if (wGradNorm > gradClip) {
            w.grad = w.grad.mul(Tensor.fromData([gradClip / wGradNorm], [1]));
          }
          w.sub_(w.grad.mul(lr));
        }

        // Clip b gradient
        if (b.grad) {
          const bGradData = await b.grad.toArray();
          const bGradNorm = Math.sqrt(
            bGradData.reduce((sum, v) => sum + v * v, 0)
          );
          if (bGradNorm > gradClip) {
            b.grad = b.grad.mul(Tensor.fromData([gradClip / bGradNorm], [1]));
          }
          b.sub_(b.grad.mul(lr));
        }

        w.grad = null;
        b.grad = null;
      });
    }

    const wFinal = await w.toArray();
    const bFinal = await b.item();

    log(
      `Final W: [${Array.from(wFinal)
        .map((v) => v.toFixed(4))
        .join(", ")}]`
    );
    log(`Expected W: [${W_true.join(", ")}]`);
    log(`Final b: ${bFinal.toFixed(4)} (Expected ${b_true})`);
    log(`Stopped at epoch ${finalEpoch}`);

    // Check convergence status
    if (!converged) {
      log("❌ Training did not converge");
      return [false, "Training diverged or did not converge"];
    }

    const wOk = wFinal.every((v, i) => Math.abs(v - W_true[i]) < 0.35);
    const bOk = Math.abs(bFinal - b_true) < 0.4;

    if (wOk && bOk) {
      log("✅ Converged successfully!");
    } else {
      log("❌ Converged but parameters not accurate enough");
    }

    return [wOk && bOk, wOk && bOk ? "Converged" : "Failed to converge"];
  });

  addInfo("Training Demo (Linear + Softmax)", async (log) => {
    log("Setting up Neural Net...");
    log("Input: [1, 4], Output: [1, 3]");

    // Input (Batch size 1)
    const x = Tensor.fromData([0.5, -0.2, 0.1, 0.8], [1, 4]);

    // Target (Class 1 is correct: [0, 1, 0])
    const target = Tensor.fromData([0, 1, 0], [1, 3]);

    // Weights & Bias
    let W = Tensor.randn([4, 3], true);
    let b = Tensor.randn([1, 3], true);

    const lr = Tensor.fromData([0.5], [1]); // Learning Rate

    log("\nStarting Training Loop (100 Steps)...");

    for (let i = 1; i <= 100; i++) {
      // Forward
      const z = x.matmul(W).add(b);
      const probs = z.softmax();

      // Loss (MSE)
      const diff = probs.sub(target);
      const loss = diff.mul(diff).mean();

      const lossVal = await loss.item();
      const probsVal = await probs.toArray();

      log(
        `Step ${i}: Loss = ${lossVal.toFixed(6)} | Probs = [${Array.from(
          probsVal
        )
          .map((p) => p.toFixed(3))
          .join(", ")}]`
      );

      // Backward
      loss.backward();

      if (!W.grad || !b.grad) {
        log("❌ Gradients missing!");
        break;
      }

      // Update (SGD)
      await noGrad(async () => {
        // W = W - lr * grad
        // In-place update
        W.sub_(W.grad!.mul(lr));
        b.sub_(b.grad!.mul(lr));

        // Zero gradients for next step
        W.grad = null;
        b.grad = null;
      });
    }

    log("\nTraining Complete!");
  });
}

runTests().catch(console.error);
