import { Tensor, noGrad } from "../src/index";
import { addTest, addInfo } from "./testUtils";

async function runTests() {
  await addTest("Initialization", async () => {
    await Tensor.init(4);
    return [true, "Initialized with 4 threads"];
  });

  await addTest("Basic Arithmetic (Add)", async () => {
    const a = Tensor.fromData([1, 2, 3], [3]);
    const b = Tensor.fromData([4, 5, 6], [3]);
    const c = a.add(b);
    const res = await c.toArray();

    const expected = [5, 7, 9];
    const match = res.every((val, i) => Math.abs(val - expected[i]) < 1e-5);

    return [
      match,
      match ? `Got [${res}]` : `Expected [${expected}], got [${res}]`,
    ];
  });

  await addTest("Matrix Multiplication", async () => {
    const M = 10,
      K = 20,
      N = 10;
    const m1 = Tensor.randn([M, K]);
    const m2 = Tensor.randn([K, N]);
    const m3 = m1.matmul(m2);

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

  await addTest("Get/Set Value", async () => {
    const t = Tensor.zeros([2, 2]);
    t.setValue([0, 1], 42);
    const val = await t.getValue([0, 1]);

    return [val === 42, `Expected 42, got ${val}`];
  });

  await addTest("Autograd (y = x^2 + 3x)", async () => {
    // y = x^2 + 3x
    // dy/dx = 2x + 3
    // Let x = 2. Then y = 4 + 6 = 10. dy/dx = 7.

    const x = Tensor.fromData([2], [1], true);
    const x2 = x.mul(x);
    const threeX = x.mul(Tensor.fromData([3], [1]));
    const y = x2.add(threeX);

    const yVal = await y.item();

    y.backward();

    if (!x.grad) return [false, "x.grad is null"];

    const gradVal = await x.grad.item();
    const err = Math.abs(gradVal - 7);

    return [err < 1e-4, `y=${yVal}, dy/dx=${gradVal} (Expected 7)`];
  });

  await addTest("MatMul Gradient (Dot Product)", async () => {
    // A (1x2) * B (2x1) -> C (1x1)
    const a = Tensor.fromData([1, 2], [1, 2], true);
    const b = Tensor.fromData([3, 4], [2, 1], true);
    const c = a.matmul(b);

    const cVal = await c.item();
    if (Math.abs(cVal - 11) > 1e-4)
      return [false, `Forward failed. Expected 11, got ${cVal}`];

    c.backward();

    if (!a.grad || !b.grad) return [false, "Gradients missing"];

    const aGrad = await a.grad.toArray(); // Expected B^T = [3, 4]
    const bGrad = await b.grad.toArray(); // Expected A^T = [1, 2]

    const aOk = Math.abs(aGrad[0] - 3) < 1e-4 && Math.abs(aGrad[1] - 4) < 1e-4;
    const bOk = Math.abs(bGrad[0] - 1) < 1e-4 && Math.abs(bGrad[1] - 2) < 1e-4;

    if (!aOk) return [false, `dA failed. Expected [3, 4], got [${aGrad}]`];
    if (!bOk) return [false, `dB failed. Expected [1, 2], got [${bGrad}]`];

    return [true, "Forward and Backward correct"];
  });

  await addTest("MLP Layer (ReLU + MatMul + Bias)", async () => {
    // x: [1, 2]
    // W: [[0.1, 0.2], [-0.1, 0.3]]
    // b: [0.1, -0.1]
    // z = xW + b
    //   = [1*0.1 + 2*-0.1, 1*0.2 + 2*0.3] + [0.1, -0.1]
    //   = [-0.1, 0.8] + [0.1, -0.1]
    //   = [0.0, 0.7]
    // a = relu(z) = [0.0, 0.7]
    // Loss = sum(a) = a * [1, 1]^T = 0.7

    const x = Tensor.fromData([1, 2], [1, 2], true);
    const W = Tensor.fromData([0.1, 0.2, -0.1, 0.3], [2, 2], true);
    const b = Tensor.fromData([0.1, -0.1], [1, 2], true);

    const z = x.matmul(W).add(b);
    const a = z.relu();

    // Mock sum by matmul with ones
    const ones = Tensor.fromData([1, 1], [2, 1]);
    const loss = a.matmul(ones); // Scalar (1x1)

    const lossVal = await loss.item();
    if (Math.abs(lossVal - 0.7) > 1e-4)
      return [false, `Forward failed. Expected 0.7, got ${lossVal}`];

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

  await addTest("Reductions (Sum/Mean)", async () => {
    const a = Tensor.fromData([1, 2, 3, 4], [4], true);
    const s = a.sum();
    const m = a.mean();

    const sVal = await s.item();
    const mVal = await m.item();

    if (Math.abs(sVal - 10) > 1e-4)
      return [false, `Sum failed. Expected 10, got ${sVal}`];
    if (Math.abs(mVal - 2.5) > 1e-4)
      return [false, `Mean failed. Expected 2.5, got ${mVal}`];

    // Backward
    // d(mean)/da = 1/N = 0.25
    m.backward();

    if (!a.grad) return [false, "Grad missing"];
    const grad = await a.grad.toArray();
    const expected = [0.25, 0.25, 0.25, 0.25];

    const match = grad.every((v) => Math.abs(v - 0.25) < 1e-4);
    return [
      match,
      match ? "Passed" : `Grad failed. Expected ${expected}, got ${grad}`,
    ];
  });

  await addTest("Broadcasting Gradient (Add)", async () => {
    // x: [2, 3]
    // b: [3]
    // y = x + b
    // loss = y.sum()

    const x = Tensor.fromData([1, 1, 1, 1, 1, 1], [2, 3], true);
    const b = Tensor.fromData([2, 2, 2], [3], true);

    const y = x.add(b);
    const loss = y.sum();

    loss.backward();

    if (!x.grad || !b.grad) return [false, "Gradients missing"];

    const xGrad = await x.grad.toArray();
    const bGrad = await b.grad.toArray();

    // x.grad should be all 1s (shape [2, 3])
    const xOk = xGrad.every((v) => Math.abs(v - 1.0) < 1e-4);

    // b.grad should be all 2s (shape [3]) because it was broadcasted across dim 0 (size 2)
    const bOk = bGrad.every((v) => Math.abs(v - 2.0) < 1e-4);

    if (!xOk) return [false, `x.grad failed. Expected all 1s, got ${xGrad}`];
    if (!bOk) return [false, `b.grad failed. Expected all 2s, got ${bGrad}`];

    return [true, "Broadcasting gradients correct"];
  });

  await addTest("noGrad Mode", async () => {
    const x = Tensor.fromData([2], [1], true);

    let y: Tensor;
    noGrad(() => {
      y = x.mul(x);
    });

    // y should not require grad
    if (y!.requiresGrad) return [false, "y.requiresGrad should be false"];
    if (y!.op !== null) return [false, "y.op should be null"];

    // Backward should do nothing (or at least not crash, but since requiresGrad is false it returns early)
    y!.backward();

    if (x.grad) return [false, "x.grad should be null"];

    return [true, "noGrad works"];
  });

  await addTest("Reshape", async () => {
    const a = Tensor.fromData([1, 2, 3, 4, 5, 6], [2, 3], true);
    const b = a.reshape([3, 2]);

    const bData = await b.toArray();
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

    return [gradMatch, "Reshape forward/backward correct"];
  });

  await addTest("Broadcasting (Add)", async () => {
    // A: [2, 3]
    // B: [3] -> Broadcast to [2, 3]
    const a = Tensor.fromData([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = Tensor.fromData([10, 20, 30], [3]);

    const c = a.add(b);
    const res = await c.toArray();

    // Expected:
    // [1+10, 2+20, 3+30, 4+10, 5+20, 6+30]
    // [11, 22, 33, 14, 25, 36]
    const expected = [11, 22, 33, 14, 25, 36];

    const match = res.every((val, i) => Math.abs(val - expected[i]) < 1e-5);
    return [match, match ? "Passed" : `Expected ${expected}, got ${res}`];
  });

  await addTest("Softmax Layer", async () => {
    // Input: [1, 4]
    const xData = [0.1, 0.2, 0.3, 0.4];
    const x = Tensor.fromData(xData, [1, 4]);

    // Weights: [4, 3]
    // Let's use fixed values to verify math
    const wData = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3];
    const W = Tensor.fromData(wData, [4, 3]);

    // Bias: [1, 3]
    const bData = [0.1, 0.2, 0.3];
    const b = Tensor.fromData(bData, [1, 3]);

    // Linear Layer
    const z = x.matmul(W).add(b);

    // Manual Verification of Linear Layer
    // z[0] = (0.1*0.1 + 0.2*0.4 + 0.3*0.7 + 0.4*0.1) + 0.1 = (0.01 + 0.08 + 0.21 + 0.04) + 0.1 = 0.34 + 0.1 = 0.44
    // z[1] = (0.1*0.2 + 0.2*0.5 + 0.3*0.8 + 0.4*0.2) + 0.2 = (0.02 + 0.10 + 0.24 + 0.08) + 0.2 = 0.44 + 0.2 = 0.64
    // z[2] = (0.1*0.3 + 0.2*0.6 + 0.3*0.9 + 0.4*0.3) + 0.3 = (0.03 + 0.12 + 0.27 + 0.12) + 0.3 = 0.54 + 0.3 = 0.84

    const zData = await z.toArray();
    const expectedZ = [0.44, 0.64, 0.84];
    const zMatch = zData.every((v, i) => Math.abs(v - expectedZ[i]) < 1e-5);

    if (!zMatch)
      return [
        false,
        `Linear layer failed. Expected ${expectedZ}, got ${zData}`,
      ];

    // Softmax
    const probs = z.softmax();

    const probsData = await probs.toArray();
    const sumProbs = probsData.reduce((a, b) => a + b, 0);

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

    const lr = Tensor.fromData([0.005], [1]);
    const epochs = 1000;

    log("\nTraining...");

    for (let i = 1; i <= epochs; i++) {
      // Forward
      const y_pred = x.matmul(w).add(b);

      // Loss (MSE)
      const diff = y_pred.sub(y_true);
      const loss = diff.mul(diff).mean();

      if (i % 50 === 0 || i === 1) {
        log(`Epoch ${i}: Loss = ${(await loss.item()).toFixed(6)}`);
      }

      // Backward
      loss.backward();

      // Update
      noGrad(() => {
        if (w.grad) w.sub_(w.grad.mul(lr));
        if (b.grad) b.sub_(b.grad.mul(lr));

        // Zero grads
        w.grad = null;
        b.grad = null;
      });
    }

    const finalW = await w.item();
    const finalB = await b.item();

    log(`\nFinal w: ${finalW.toFixed(4)} (Expected ~2.0)`);
    log(`Final b: ${finalB.toFixed(4)} (Expected ~1.0)`);

    const success = Math.abs(finalW - 2) < 0.1 && Math.abs(finalB - 1) < 0.3;
    if (success) {
      log("✅ Converged successfully!");
    } else {
      log("❌ Failed to converge.");
    }

    return [success, success ? "Converged" : "Failed to converge"];
  });

  await addTest("CrossEntropy Loss", async () => {
    // Logits: [1.0, 2.0, 3.0]
    // Target: [0, 0, 1]
    const logits = Tensor.fromData([1.0, 2.0, 3.0], [1, 3], true);
    const target = Tensor.fromData([0, 0, 1], [1, 3]);

    const loss = Tensor.crossEntropy(logits, target);
    const lossVal = await loss.item();

    // Expected:
    // exp: [2.718, 7.389, 20.085]
    // sum: 30.192
    // p2: 20.085 / 30.192 = 0.6652
    // -log(0.6652) = 0.4076
    const expected = 0.4076;

    const match = Math.abs(lossVal - expected) < 1e-3;

    // Backward
    loss.backward();
    if (!logits.grad) return [false, "Grad missing"];
    
    // dLoss/dLogits = probs - target
    // p0 - 0 = 0.0900
    // p1 - 0 = 0.2447
    // p2 - 1 = 0.6652 - 1 = -0.3348
    
    const grad = await logits.grad.toArray();
    const gExpected = [0.0900, 0.2447, -0.3348];
    
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

    const lr = Tensor.fromData([0.01], [1]);
    const epochs = 500;

    log("Training...");

    for (let i = 1; i <= epochs; i++) {
      const y_pred = x.matmul(w).add(b);
      const diff = y_pred.sub(y_true);
      const loss = diff.mul(diff).mean();

      if (i % 100 === 0) {
        log(`Epoch ${i}: Loss = ${(await loss.item()).toFixed(6)}`);
      }

      loss.backward();

      noGrad(() => {
        if (w.grad) w.sub_(w.grad.mul(lr));
        if (b.grad) b.sub_(b.grad.mul(lr));
        w.grad = null;
        b.grad = null;
      });
    }

    const wFinal = await w.toArray();
    const bFinal = await b.item();

    log(`Final W: [${Array.from(wFinal).map((v) => v.toFixed(4)).join(", ")}]`);
    log(`Expected W: [${W_true.join(", ")}]`);
    log(`Final b: ${bFinal.toFixed(4)} (Expected ${b_true})`);

    const wOk = wFinal.every((v, i) => Math.abs(v - W_true[i]) < 0.2);
    const bOk = Math.abs(bFinal - b_true) < 0.2;

    return [wOk && bOk, wOk && bOk ? "Converged" : "Failed to converge"];
  });

  await addInfo("Training Demo (Linear + Softmax)", async (log) => {
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
      // W = W - lr * grad
      // In-place update
      W.sub_(W.grad.mul(lr));
      b.sub_(b.grad.mul(lr));

      // Zero gradients for next step
      W.grad = null;
      b.grad = null;
    }

    log("\nTraining Complete!");
  });
}

runTests().catch(console.error);
