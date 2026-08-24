import { Tensor, init, shutdown } from "../../src/index";
import { defineTest } from "../framework/define";
import type { RunContext } from "../framework/types";

type Backend = "workers" | "webgpu" | "wasm";

const B = 1;
const HIDDEN = 256;
const IN_SIZE = 640;

function fill(shape: number[], seed: number, scale = 0.1): Float32Array {
  const n = shape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sin((i + seed) * 0.017) * scale;
  return out;
}

function composedStep(
  x: Tensor,
  h: Tensor,
  c: Tensor,
  wIh: Tensor,
  wHh: Tensor,
  bIh: Tensor,
  bHh: Tensor,
  hidden: number,
): [Tensor, Tensor] {
  const B_ = x.shape[0];
  const H = hidden;
  const gates = x
    .matmul(wIh.transpose())
    .add(bIh)
    .add(h.matmul(wHh.transpose()))
    .add(bHh);
  const iG = gates.slice([[0, B_], [0, H]]).sigmoid();
  const fG = gates.slice([[0, B_], [H, 2 * H]]).sigmoid();
  const gC = gates.slice([[0, B_], [2 * H, 3 * H]]).tanh();
  const oG = gates.slice([[0, B_], [3 * H, 4 * H]]).sigmoid();
  const cNew = fG.mul(c).add(iG.mul(gC));
  const hNew = oG.mul(cNew.tanh());
  return [hNew, cNew];
}

async function runBoth(backend: Backend) {
  await init({ backend, memorySizeMB: 64 });

  const x = Tensor.fromData(Array.from(fill([B, IN_SIZE], 0)), [B, IN_SIZE]);
  const h = Tensor.fromData(Array.from(fill([B, HIDDEN], 7, 0.2)), [B, HIDDEN]);
  const c = Tensor.fromData(Array.from(fill([B, HIDDEN], 13, 0.15)), [B, HIDDEN]);
  const wIh = Tensor.fromData(Array.from(fill([4 * HIDDEN, IN_SIZE], 3, 0.05)), [4 * HIDDEN, IN_SIZE]);
  const wHh = Tensor.fromData(Array.from(fill([4 * HIDDEN, HIDDEN], 5, 0.05)), [4 * HIDDEN, HIDDEN]);
  const bIh = Tensor.fromData(Array.from(fill([4 * HIDDEN], 11, 0.01)), [4 * HIDDEN]);
  const bHh = Tensor.fromData(Array.from(fill([4 * HIDDEN], 17, 0.01)), [4 * HIDDEN]);

  const [hRef, cRef] = composedStep(x, h, c, wIh, wHh, bIh, bHh, HIDDEN);
  const refH = await hRef.toArray();
  const refC = await cRef.toArray();

  const packed = x.lstmStep(h, c, wIh, wHh, bIh, bHh);
  const hFused = packed.slice([[0, B], [0, HIDDEN]]);
  const cFused = packed.slice([[0, B], [HIDDEN, 2 * HIDDEN]]);
  const outH = await hFused.toArray();
  const outC = await cFused.toArray();

  shutdown();
  return { refH, refC, outH, outC };
}

function maxDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

async function runParity(backend: Backend, { log }: RunContext) {
  log(`--- LSTM_STEP fused vs composed on ${backend} (B=${B}, hidden=${HIDDEN}, in=${IN_SIZE}) ---`);
  const { refH, refC, outH, outC } = await runBoth(backend);
  const dH = maxDiff(refH, outH);
  const dC = maxDiff(refC, outC);
  const tol = 5e-5;
  log(`  h_new max|composed - fused| = ${dH.toExponential(3)}   tol ${tol.toExponential(1)}`);
  log(`  c_new max|composed - fused| = ${dC.toExponential(3)}   tol ${tol.toExponential(1)}`);
  const ok = dH <= tol && dC <= tol;
  return { pass: ok, message: ok ? `${backend} ok` : `${backend} diverged` };
}

defineTest({
  name: "LSTM_STEP: fused vs composed",
  params: ["workers", "webgpu", "wasm"] as Backend[],
  description:
    "One LSTM step from x, h, c and PyTorch-layout weights. Fused kernel (one dispatch) vs composed primitives (matmul+add+slice+sigmoid+tanh+mul chain). fp32 tolerance.",
  runner: runParity,
});
