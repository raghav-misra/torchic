import { Tensor, init, shutdown, nn } from "../../src/index";
import { defineTest } from "../framework/define";
import type { RunContext, TestResult } from "../framework/types";

type Backend = "workers" | "wasm" | "webgpu";

// Scalar JS reference implementations for verification.
function refConv1d(
  x: number[],
  w: number[],
  bias: number[] | null,
  B: number,
  Cin: number,
  Lin: number,
  Cout: number,
  K: number,
  stride: number,
  pad: number,
  dil: number,
): { out: number[]; Lout: number } {
  const Lout = Math.floor((Lin + 2 * pad - dil * (K - 1) - 1) / stride) + 1;
  const out = new Array(B * Cout * Lout).fill(0);
  for (let b = 0; b < B; b++) {
    for (let co = 0; co < Cout; co++) {
      const biasVal = bias ? bias[co] : 0;
      for (let lo = 0; lo < Lout; lo++) {
        let sum = biasVal;
        for (let ci = 0; ci < Cin; ci++) {
          for (let k = 0; k < K; k++) {
            const li = lo * stride + k * dil - pad;
            if (li >= 0 && li < Lin) {
              sum += x[b * Cin * Lin + ci * Lin + li] * w[co * Cin * K + ci * K + k];
            }
          }
        }
        out[b * Cout * Lout + co * Lout + lo] = sum;
      }
    }
  }
  return { out, Lout };
}

function refConvTranspose1d(
  x: number[],
  w: number[],
  bias: number[] | null,
  B: number,
  Cin: number,
  Lin: number,
  Cout: number,
  K: number,
  stride: number,
  pad: number,
  dil: number,
  outPad: number,
): { out: number[]; Lout: number } {
  const Lout = (Lin - 1) * stride - 2 * pad + dil * (K - 1) + outPad + 1;
  const out = new Array(B * Cout * Lout).fill(0);
  for (let b = 0; b < B; b++) {
    for (let co = 0; co < Cout; co++) {
      const biasVal = bias ? bias[co] : 0;
      for (let lo = 0; lo < Lout; lo++) out[b * Cout * Lout + co * Lout + lo] = biasVal;
    }
    for (let ci = 0; ci < Cin; ci++) {
      for (let li = 0; li < Lin; li++) {
        const xv = x[b * Cin * Lin + ci * Lin + li];
        for (let co = 0; co < Cout; co++) {
          for (let k = 0; k < K; k++) {
            const lo = li * stride + k * dil - pad;
            if (lo >= 0 && lo < Lout) {
              out[b * Cout * Lout + co * Lout + lo] += xv * w[ci * Cout * K + co * K + k];
            }
          }
        }
      }
    }
  }
  return { out, Lout };
}

function fill(n: number, seed: number): number[] {
  const r: number[] = [];
  let x = seed;
  for (let i = 0; i < n; i++) {
    x = (Math.imul(1103515245, x) + 12345) | 0;
    r.push(((x & 0xffff) - 0x8000) / 0x8000);
  }
  return r;
}

async function runConv(backend: Backend, { log }: RunContext): Promise<TestResult> {
  const threadCount = backend === "webgpu" ? 1 : 4;
  await init({ backend, threadCount, memorySizeMB: 32 });
  try {
    const checks: { name: string; ok: boolean; detail: string }[] = [];
    const check = (n: string, ok: boolean, d = "") => checks.push({ name: n, ok, detail: d });

    // Conv1D correctness cases.
    const conv1dCases = [
      { B: 1, Cin: 1, Lin: 8, Cout: 1, K: 3, stride: 1, pad: 0, dil: 1 },
      { B: 2, Cin: 3, Lin: 16, Cout: 5, K: 3, stride: 1, pad: 1, dil: 1 },
      { B: 1, Cin: 4, Lin: 32, Cout: 8, K: 5, stride: 2, pad: 2, dil: 1 },
      { B: 1, Cin: 2, Lin: 20, Cout: 4, K: 3, stride: 1, pad: 2, dil: 2 }, // dilated
      { B: 2, Cin: 8, Lin: 16, Cout: 16, K: 3, stride: 1, pad: 1, dil: 1, noBias: true },
    ];

    for (const c of conv1dCases) {
      const xData = fill(c.B * c.Cin * c.Lin, c.Cin * 31 + c.Lin);
      const wData = fill(c.Cout * c.Cin * c.K, c.Cout * 17 + c.K);
      const bData = c.noBias ? null : fill(c.Cout, c.Cout * 11);

      const x = Tensor.fromData(xData, [c.B, c.Cin, c.Lin]);
      const w = Tensor.fromData(wData, [c.Cout, c.Cin, c.K]);
      const bias = bData ? Tensor.fromData(bData) : null;

      const y = x.conv1d(w, bias, { stride: c.stride, padding: c.pad, dilation: c.dil });
      const out = await y.toArray();
      const { out: ref, Lout } = refConv1d(
        xData,
        wData,
        bData,
        c.B,
        c.Cin,
        c.Lin,
        c.Cout,
        c.K,
        c.stride,
        c.pad,
        c.dil,
      );

      let maxErr = 0;
      for (let i = 0; i < out.length; i++) maxErr = Math.max(maxErr, Math.abs(out[i] - ref[i]));
      const shapeOk = y.shape[0] === c.B && y.shape[1] === c.Cout && y.shape[2] === Lout;
      const ok = shapeOk && maxErr < 1e-5;
      check(
        `Conv1D B=${c.B} Cin=${c.Cin} Lin=${c.Lin} Cout=${c.Cout} K=${c.K} s=${c.stride} p=${c.pad} d=${c.dil}${c.noBias ? " nb" : ""}`,
        ok,
        `Lout=${Lout} maxErr=${maxErr.toExponential(2)}`,
      );
      log(`  ${ok ? "OK  " : "FAIL"} Conv1D shape=[${y.shape}]  maxErr=${maxErr.toExponential(2)}`);
    }

    // ConvTranspose1D correctness cases.
    const ctCases = [
      { B: 1, Cin: 1, Lin: 4, Cout: 1, K: 3, stride: 1, pad: 0, dil: 1, outPad: 0 },
      { B: 2, Cin: 3, Lin: 8, Cout: 5, K: 4, stride: 2, pad: 1, dil: 1, outPad: 0 },
      { B: 1, Cin: 4, Lin: 5, Cout: 2, K: 5, stride: 3, pad: 2, dil: 1, outPad: 1 }, // upsample
      { B: 1, Cin: 2, Lin: 6, Cout: 3, K: 3, stride: 1, pad: 0, dil: 2, outPad: 0 }, // dilated
    ];

    for (const c of ctCases) {
      const xData = fill(c.B * c.Cin * c.Lin, c.Cin * 41 + c.Lin);
      const wData = fill(c.Cin * c.Cout * c.K, c.Cout * 23 + c.K);
      const bData = fill(c.Cout, c.Cout * 7);

      const x = Tensor.fromData(xData, [c.B, c.Cin, c.Lin]);
      const w = Tensor.fromData(wData, [c.Cin, c.Cout, c.K]);
      const bias = Tensor.fromData(bData);

      const y = x.convTranspose1d(w, bias, {
        stride: c.stride,
        padding: c.pad,
        dilation: c.dil,
        outputPadding: c.outPad,
      });
      const out = await y.toArray();
      const { out: ref, Lout } = refConvTranspose1d(
        xData,
        wData,
        bData,
        c.B,
        c.Cin,
        c.Lin,
        c.Cout,
        c.K,
        c.stride,
        c.pad,
        c.dil,
        c.outPad,
      );

      let maxErr = 0;
      for (let i = 0; i < out.length; i++) maxErr = Math.max(maxErr, Math.abs(out[i] - ref[i]));
      const shapeOk = y.shape[0] === c.B && y.shape[1] === c.Cout && y.shape[2] === Lout;
      const ok = shapeOk && maxErr < 1e-5;
      check(
        `ConvT1D B=${c.B} Cin=${c.Cin} Lin=${c.Lin} Cout=${c.Cout} K=${c.K} s=${c.stride} p=${c.pad} d=${c.dil} op=${c.outPad}`,
        ok,
        `Lout=${Lout} maxErr=${maxErr.toExponential(2)}`,
      );
      log(`  ${ok ? "OK  " : "FAIL"} ConvT1D shape=[${y.shape}]  maxErr=${maxErr.toExponential(2)}`);
    }

    // Module wrappers: nn.Conv1d and nn.ConvTranspose1d should train (init defaults + forward).
    {
      const x = Tensor.randn([2, 16, 32]);
      const conv = new nn.Conv1d(16, 32, 5, { padding: 2 });
      const y = conv.forward(x);
      const yData = await y.toArray();
      const shapeOk = y.shape[0] === 2 && y.shape[1] === 32 && y.shape[2] === 32;
      const finite = Array.from(yData).every((v) => Number.isFinite(v));
      check("nn.Conv1d module", shapeOk && finite, `shape=${y.shape}`);
      log(`  ${shapeOk && finite ? "OK  " : "FAIL"} nn.Conv1d shape=[${y.shape}]`);
    }

    {
      const x = Tensor.randn([1, 8, 16]);
      const upsample = new nn.ConvTranspose1d(8, 4, 4, { stride: 2, padding: 1 });
      const y = upsample.forward(x);
      const shapeOk = y.shape[0] === 1 && y.shape[1] === 4;
      check("nn.ConvTranspose1d module", shapeOk, `shape=${y.shape}`);
      log(`  ${shapeOk ? "OK  " : "FAIL"} nn.ConvTranspose1d shape=[${y.shape}]`);
    }

    const failed = checks.filter((c) => !c.ok);
    for (const c of checks) {
      log(`  [${c.ok ? "OK  " : "FAIL"}] ${c.name}${c.detail ? " — " + c.detail : ""}`);
    }
    return {
      pass: failed.length === 0,
      message:
        failed.length === 0
          ? `${checks.length}/${checks.length} checks passed`
          : `${failed.length}/${checks.length} failed`,
    };
  } finally {
    shutdown();
  }
}

defineTest<Backend>({
  name: "Conv1D + ConvTranspose1D: cross-backend correctness",
  paramName: "backend",
  params: ["workers", "wasm", "webgpu"],
  description:
    "1-D conv and transposed conv on all three backends, checked against a scalar JS reference.",
  runner: runConv,
});
