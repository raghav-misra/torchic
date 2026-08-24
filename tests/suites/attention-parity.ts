import { Tensor, init, shutdown } from "../../src/index";
import { defineTest } from "../framework/define";
import type { RunContext } from "../framework/types";

type Backend = "workers" | "webgpu";

const B = 1;
const NH = 12;
const DH = 64;
const H = NH * DH;

interface AttnOutputs {
  qkOut: Float32Array;
  softmaxOut: Float32Array;
  contextOut: Float32Array;
  finalOut: Float32Array;
}

function splitHeads(t: Tensor, T: number): Tensor {
  return t.reshape([B, T, NH, DH]).transpose(1, 2).reshape([B * NH, T, DH]);
}

// Reproduces the AlbertAttention path (minus dense/residual/LN) for a fixed T.
async function runAttention(backend: Backend, T: number): Promise<AttnOutputs> {
  await init({ backend, memorySizeMB: 128 });

  const xData = new Float32Array(B * T * H);
  for (let i = 0; i < xData.length; i++) xData[i] = Math.sin(i * 0.017);
  const wqData = new Float32Array(H * H);
  const wkData = new Float32Array(H * H);
  const wvData = new Float32Array(H * H);
  const wdData = new Float32Array(H * H);
  for (let i = 0; i < wqData.length; i++) {
    wqData[i] = Math.sin(i * 0.011) * 0.05;
    wkData[i] = Math.cos(i * 0.013) * 0.05;
    wvData[i] = Math.sin(i * 0.019) * 0.05;
    wdData[i] = Math.cos(i * 0.021) * 0.05;
  }

  const x = Tensor.fromData(Array.from(xData), [B, T, H]);
  const wq = Tensor.fromData(Array.from(wqData), [H, H]);
  const wk = Tensor.fromData(Array.from(wkData), [H, H]);
  const wv = Tensor.fromData(Array.from(wvData), [H, H]);
  const wd = Tensor.fromData(Array.from(wdData), [H, H]);
  const scale = Tensor.fromData([1 / Math.sqrt(DH)]);

  const q = splitHeads(x.reshape([-1, H]).matmul(wq.transpose()).reshape([B, T, H]), T);
  const k = splitHeads(x.reshape([-1, H]).matmul(wk.transpose()).reshape([B, T, H]), T);
  const v = splitHeads(x.reshape([-1, H]).matmul(wv.transpose()).reshape([B, T, H]), T);

  const kT = k.transpose(-1, -2);
  const qk = q.bmm(kT).mul(scale);
  const qkOut = await qk.toArray();

  const flat = qk.reshape([B * NH * T, T]).softmax(-1).reshape([B * NH, T, T]);
  const softmaxOut = await flat.toArray();

  const av = flat.bmm(v);
  const context = av.reshape([B, NH, T, DH]).transpose(1, 2).reshape([B, T, H]);
  const contextOut = await context.toArray();

  const projected = context.reshape([-1, H]).matmul(wd.transpose()).reshape([B, T, H]);
  const finalOut = await projected.toArray();

  shutdown();
  return { qkOut, softmaxOut, contextOut, finalOut };
}

function stats(x: Float32Array): { mean: number; std: number; min: number; max: number } {
  let s = 0, sq = 0, mn = Infinity, mx = -Infinity;
  for (const v of x) { s += v; sq += v * v; if (v < mn) mn = v; if (v > mx) mx = v; }
  const mean = s / x.length;
  return { mean, std: Math.sqrt(Math.max(0, sq / x.length - mean * mean)), min: mn, max: mx };
}

function maxDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

async function runParity(T: number, { log }: RunContext) {
  log(`--- attention parity at T=${T} ---`);
  log("workers backend");
  const w = await runAttention("workers", T);
  log("webgpu backend");
  const g = await runAttention("webgpu", T);

  const rows: [keyof AttnOutputs, number][] = [
    ["qkOut", 5e-4],
    ["softmaxOut", 1e-5],
    ["contextOut", 1e-4],
    ["finalOut", 1e-3],
  ];
  let ok = true;
  for (const [name, tol] of rows) {
    const diff = maxDiff(w[name], g[name]);
    const sw = stats(w[name]);
    const sg = stats(g[name]);
    const pass = diff <= tol;
    if (!pass) ok = false;
    log(
      `  ${name.padEnd(11)} max|w-g|=${diff.toExponential(3)}   tol ${tol.toExponential(1)}   ${pass ? "OK" : "FAIL"}`,
    );
    log(
      `    workers std=${sw.std.toFixed(6)} max=${sw.max.toFixed(4)}   webgpu std=${sg.std.toFixed(6)} max=${sg.max.toFixed(4)}`,
    );
  }
  return { pass: ok, message: ok ? `T=${T} ok` : `T=${T} diverged` };
}

defineTest({
  name: "Attention parity (WebGPU vs Workers)",
  params: [15, 50],
  description:
    "Runs one Albert-style attention block on both backends with identical synthetic inputs. If T=15 matches but T=50 diverges, the divergence is inside a length-dependent op on WebGPU.",
  runner: runParity,
});
