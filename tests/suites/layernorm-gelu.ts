import { Tensor, init, shutdown, noGrad, nn } from "../../src/index";
import { defineTest } from "../framework/define";
import type { RunContext, TestResult } from "../framework/types";

type Backend = "workers" | "wasm" | "webgpu";

// LayerNorm + GELU end-to-end across all three backends. Validates composed
// forward numerics, autograd flow, and finite-difference grads.

interface Check {
  name: string;
  ok: boolean;
  detail: string;
}

function check(name: string, ok: boolean, detail = ""): Check {
  return { name, ok, detail };
}

async function runChecks(backend: Backend, { log }: RunContext): Promise<TestResult> {
  const threads = backend === "webgpu" ? 1 : 2;
  await init({ backend, threadCount: threads, memorySizeMB: 32 });
  try {
    const checks: Check[] = [];

    // 1. LayerNorm forward: with default weight=1, bias=0, each row should have
    //    mean ≈ 0 and variance ≈ 1 (biased estimator, matches PyTorch).
    {
      const B = 4;
      const D = 32;
      const x = Tensor.randn([B, D]);
      const ln = new nn.LayerNorm(D);
      const y = ln.forward(x);
      const yData = await y.toArray();

      let maxMean = 0;
      let maxVarErr = 0;
      for (let r = 0; r < B; r++) {
        let mean = 0;
        for (let c = 0; c < D; c++) mean += yData[r * D + c];
        mean /= D;
        let variance = 0;
        for (let c = 0; c < D; c++) {
          const d = yData[r * D + c] - mean;
          variance += d * d;
        }
        variance /= D;
        maxMean = Math.max(maxMean, Math.abs(mean));
        maxVarErr = Math.max(maxVarErr, Math.abs(variance - 1));
      }
      log(`  LN default: |mean| max=${maxMean.toExponential(2)}  |var-1| max=${maxVarErr.toExponential(2)}`);
      checks.push(check("LN default rows have mean≈0", maxMean < 1e-4, `max=${maxMean}`));
      checks.push(check("LN default rows have var≈1", maxVarErr < 1e-2, `max=${maxVarErr}`));
    }

    // 2. LayerNorm with affine: set weight=2, bias=3, output should be 2*xhat + 3.
    {
      const D = 16;
      const x = Tensor.randn([2, D]);
      const ln = new nn.LayerNorm(D);
      await noGrad(async () => {
        const twos = Tensor.fromData(new Array(D).fill(2));
        const threes = Tensor.fromData(new Array(D).fill(3));
        ln.weight.write(await twos.toArray());
        ln.bias.write(await threes.toArray());
      });
      const y = ln.forward(x);
      const yData = await y.toArray();

      // Recompute xhat from x manually and check y = 2*xhat + 3.
      const xData = await x.toArray();
      let maxErr = 0;
      for (let r = 0; r < 2; r++) {
        let mean = 0;
        for (let c = 0; c < D; c++) mean += xData[r * D + c];
        mean /= D;
        let variance = 0;
        for (let c = 0; c < D; c++) {
          const d = xData[r * D + c] - mean;
          variance += d * d;
        }
        variance /= D;
        const invStd = 1 / Math.sqrt(variance + 1e-5);
        for (let c = 0; c < D; c++) {
          const xhat = (xData[r * D + c] - mean) * invStd;
          const expected = 2 * xhat + 3;
          maxErr = Math.max(maxErr, Math.abs(yData[r * D + c] - expected));
        }
      }
      log(`  LN affine: max |actual - (2*xhat+3)| = ${maxErr.toExponential(2)}`);
      checks.push(check("LN affine matches reference", maxErr < 1e-4, `max=${maxErr}`));
    }

    // 3. GELU forward matches the tanh-approximation reference.
    {
      const xData = [-3, -1, -0.5, 0, 0.5, 1, 2, 3];
      const x = Tensor.fromData(xData);
      const y = x.gelu();
      const yData = await y.toArray();
      const c = Math.sqrt(2 / Math.PI);
      const refGelu = (v: number) => 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
      let maxErr = 0;
      for (let i = 0; i < xData.length; i++) {
        maxErr = Math.max(maxErr, Math.abs(yData[i] - refGelu(xData[i])));
      }
      log(`  GELU forward: max err = ${maxErr.toExponential(2)}`);
      checks.push(check("GELU matches tanh approx", maxErr < 1e-5, `max=${maxErr}`));
    }

    // 4. GELU autograd via finite differences. We differentiate sum(gelu(x)) so
    //    each x_i's grad is d(gelu)/dx.
    {
      const x = Tensor.fromData([-2, -0.5, 0.5, 2], undefined, true);
      const y = x.gelu().sum();
      y.backward();
      const grads = await x.grad!.toArray();

      const xData = await x.toArray();
      const eps = 1e-3;
      const c = Math.sqrt(2 / Math.PI);
      const refGelu = (v: number) => 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
      let maxErr = 0;
      for (let i = 0; i < xData.length; i++) {
        const numeric = (refGelu(xData[i] + eps) - refGelu(xData[i] - eps)) / (2 * eps);
        const err = Math.abs(grads[i] - numeric);
        maxErr = Math.max(maxErr, err);
        log(`    x=${xData[i].toFixed(2)}  analytic=${grads[i].toFixed(5)}  numeric=${numeric.toFixed(5)}`);
      }
      checks.push(check("GELU grad matches finite diff", maxErr < 1e-3, `max=${maxErr}`));
    }

    // 5. LayerNorm autograd end-to-end. Note: sum(LN(x)) has zero grad w.r.t. x
    //    because LN is translation-invariant, so use an L2 loss instead. LN
    //    forces sum_i xhat_i = 0, so the interesting signal is in the affine
    //    params (weight, bias).
    {
      const B = 3;
      const D = 8;
      const x = Tensor.randn([B, D], true);
      const ln = new nn.LayerNorm(D);
      const y = ln.forward(x);
      const loss = y.mul(y).sum();
      loss.backward();
      const gx = await x.grad!.toArray();
      const gw = await ln.weight.grad!.toArray();
      const gb = await ln.bias.grad!.toArray();

      const allFinite = (arr: Float32Array) => Array.from(arr).every((v) => Number.isFinite(v));
      const anyNonZero = (arr: Float32Array) => Array.from(arr).some((v) => Math.abs(v) > 1e-6);

      checks.push(check("LN dx finite", allFinite(gx)));
      checks.push(check("LN dx non-zero (L2 loss)", anyNonZero(gx)));
      checks.push(check("LN dweight finite + non-zero", allFinite(gw) && anyNonZero(gw)));
      checks.push(check("LN dbias finite + non-zero", allFinite(gb) && anyNonZero(gb)));
      log(`  LN grad shapes: dx=[${B},${D}]  dweight=[${gw.length}]  dbias=[${gb.length}]`);
    }

    const failed = checks.filter((c) => !c.ok);
    for (const c of checks) log(`  [${c.ok ? "OK  " : "FAIL"}] ${c.name}${c.detail ? " — " + c.detail : ""}`);
    if (failed.length > 0) {
      return { pass: false, message: `${failed.length}/${checks.length} failed: ${failed.map((c) => c.name).join(", ")}` };
    }
    return { pass: true, message: `${checks.length}/${checks.length} checks passed` };
  } finally {
    shutdown();
  }
}

defineTest<Backend>({
  name: "nn.LayerNorm + GELU: cross-backend correctness",
  paramName: "backend",
  params: ["workers", "wasm", "webgpu"],
  description:
    "Validates the composed nn.LayerNorm module, GELU forward/backward, and autograd through the new sqrt/rsqrt primitives on every backend.",
  runner: async (backend, ctx) => runChecks(backend, ctx),
});
