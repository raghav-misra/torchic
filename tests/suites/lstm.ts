import { Tensor, init, shutdown, nn } from "../../src/index";
import { defineTest } from "../framework/define";
import type { RunContext, TestResult } from "../framework/types";

type Backend = "workers" | "wasm" | "webgpu";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// Scalar reference: PyTorch-layout LSTMCell forward.
function refLSTMCell(
  x: number[],
  h: number[],
  c: number[],
  wih: number[],
  whh: number[],
  bih: number[],
  bhh: number[],
  B: number,
  I: number,
  H: number,
): { hNew: number[]; cNew: number[] } {
  // gates[b, o] = sum_i x[b,i] * wih[o,i] + bih[o] + sum_j h[b,j] * whh[o,j] + bhh[o]
  const gates = new Array(B * 4 * H).fill(0);
  for (let b = 0; b < B; b++) {
    for (let o = 0; o < 4 * H; o++) {
      let s = bih[o] + bhh[o];
      for (let i = 0; i < I; i++) s += x[b * I + i] * wih[o * I + i];
      for (let j = 0; j < H; j++) s += h[b * H + j] * whh[o * H + j];
      gates[b * 4 * H + o] = s;
    }
  }
  const hNew = new Array(B * H).fill(0);
  const cNew = new Array(B * H).fill(0);
  for (let b = 0; b < B; b++) {
    for (let k = 0; k < H; k++) {
      const gi = sigmoid(gates[b * 4 * H + 0 * H + k]);
      const gf = sigmoid(gates[b * 4 * H + 1 * H + k]);
      const gg = Math.tanh(gates[b * 4 * H + 2 * H + k]);
      const go = sigmoid(gates[b * 4 * H + 3 * H + k]);
      const cNext = gf * c[b * H + k] + gi * gg;
      cNew[b * H + k] = cNext;
      hNew[b * H + k] = go * Math.tanh(cNext);
    }
  }
  return { hNew, cNew };
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

async function runLSTM(backend: Backend, { log }: RunContext): Promise<TestResult> {
  const threads = backend === "webgpu" ? 1 : 4;
  await init({ backend, threadCount: threads, memorySizeMB: 32 });
  try {
    const checks: { name: string; ok: boolean; detail: string }[] = [];
    const check = (name: string, ok: boolean, detail = "") => checks.push({ name, ok, detail });

    // Correctness cases.
    const cases = [
      { B: 1, I: 4, H: 8, seed: 1 },
      { B: 3, I: 16, H: 32, seed: 7 },
      { B: 2, I: 8, H: 16, seed: 42 },
    ];

    for (const c of cases) {
      const xData = fill(c.B * c.I, c.seed + 1);
      const hData = fill(c.B * c.H, c.seed + 2);
      const cData = fill(c.B * c.H, c.seed + 3);
      const wihData = fill(4 * c.H * c.I, c.seed + 4);
      const whhData = fill(4 * c.H * c.H, c.seed + 5);
      const bihData = fill(4 * c.H, c.seed + 6);
      const bhhData = fill(4 * c.H, c.seed + 7);

      const cell = new nn.LSTMCell(c.I, c.H);
      // Overwrite the random init with our fixture weights so the reference and
      // torchic see identical inputs.
      await cell.load_state_dict({
        weight_ih: Tensor.fromData(wihData, [4 * c.H, c.I]),
        weight_hh: Tensor.fromData(whhData, [4 * c.H, c.H]),
        bias_ih: Tensor.fromData(bihData),
        bias_hh: Tensor.fromData(bhhData),
      });

      const x = Tensor.fromData(xData, [c.B, c.I]);
      const h = Tensor.fromData(hData, [c.B, c.H]);
      const cSt = Tensor.fromData(cData, [c.B, c.H]);
      const [hNew, cNew] = cell.forward(x, [h, cSt]);
      const hOut = await hNew.toArray();
      const cOut = await cNew.toArray();

      const ref = refLSTMCell(xData, hData, cData, wihData, whhData, bihData, bhhData, c.B, c.I, c.H);
      let hErr = 0;
      let cErr = 0;
      for (let i = 0; i < hOut.length; i++) hErr = Math.max(hErr, Math.abs(hOut[i] - ref.hNew[i]));
      for (let i = 0; i < cOut.length; i++) cErr = Math.max(cErr, Math.abs(cOut[i] - ref.cNew[i]));
      const ok = hErr < 1e-4 && cErr < 1e-4;
      check(
        `LSTMCell B=${c.B} I=${c.I} H=${c.H}`,
        ok,
        `hErr=${hErr.toExponential(2)} cErr=${cErr.toExponential(2)}`,
      );
      log(`  ${ok ? "OK  " : "FAIL"} B=${c.B} I=${c.I} H=${c.H}  hErr=${hErr.toExponential(2)}  cErr=${cErr.toExponential(2)}`);
    }

    // Multi-step unroll: run cell 4 times, verify final (h, c) are finite and shape-preserving.
    {
      const cell = new nn.LSTMCell(8, 16);
      let [h, c] = cell.initialState(2);
      for (let t = 0; t < 4; t++) {
        const xt = Tensor.randn([2, 8]);
        [h, c] = cell.forward(xt, [h, c]);
      }
      const hData = await h.toArray();
      const cData = await c.toArray();
      const finite =
        Array.from(hData).every((v) => Number.isFinite(v)) &&
        Array.from(cData).every((v) => Number.isFinite(v));
      const shapeOk = h.shape[0] === 2 && h.shape[1] === 16 && c.shape[0] === 2 && c.shape[1] === 16;
      check("LSTMCell 4-step unroll", finite && shapeOk, `hShape=${h.shape}`);
      log(`  ${finite && shapeOk ? "OK  " : "FAIL"} 4-step unroll  hShape=[${h.shape}]  cShape=[${c.shape}]`);
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
  name: "LSTMCell: cross-backend correctness",
  paramName: "backend",
  params: ["workers", "wasm", "webgpu"],
  description:
    "PyTorch-layout LSTMCell (weight_ih, weight_hh, bias_ih, bias_hh) composed from matmul + sigmoid + tanh + slice, checked against a scalar JS reference on every backend.",
  runner: runLSTM,
});
