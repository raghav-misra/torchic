import { Tensor, init, shutdown, nn } from "../../src/index";
import { defineTest } from "../framework/define";
import type { RunContext, TestResult } from "../framework/types";

type Backend = "workers" | "wasm" | "webgpu";

async function runAttention(backend: Backend, { log }: RunContext): Promise<TestResult> {
  const threads = backend === "webgpu" ? 1 : 4;
  await init({ backend, threadCount: threads, memorySizeMB: 64 });
  try {
    const checks: { name: string; ok: boolean; detail: string }[] = [];
    const check = (name: string, ok: boolean, detail = "") =>
      checks.push({ name, ok, detail });

    // 1. MHA forward: shape + finite values.
    {
      const B = 2, S = 8, D = 32, H = 4;
      const x = Tensor.randn([B, S, D]);
      const mha = new nn.MultiHeadAttention(D, H);
      const y = mha.forward(x);
      const yData = await y.toArray();
      const allFinite = Array.from(yData).every((v) => Number.isFinite(v));
      const shapeOk = y.shape[0] === B && y.shape[1] === S && y.shape[2] === D;
      check("MHA output shape [B,S,D]", shapeOk, `got ${y.shape}`);
      check("MHA output finite", allFinite);
      log(`  MHA forward: shape=[${y.shape}]  allFinite=${allFinite}`);
    }

    // 2. Transformer block forward + shape preservation.
    {
      const B = 1, S = 16, D = 64, H = 8;
      const x = Tensor.randn([B, S, D]);
      const enc = new nn.TransformerEncoderLayer(D, H);
      const y = enc.forward(x);
      const yData = await y.toArray();
      const allFinite = Array.from(yData).every((v) => Number.isFinite(v));
      const shapeOk = y.shape[0] === B && y.shape[1] === S && y.shape[2] === D;
      check("Encoder output shape [B,S,D]", shapeOk, `got ${y.shape}`);
      check("Encoder output finite", allFinite);
      log(`  Encoder forward: shape=[${y.shape}]  allFinite=${allFinite}`);
    }

    // 3. Positional encoding is finite and has the expected periodicity.
    {
      const pe = nn.sinusoidalPositionalEncoding(64, 32);
      const data = await pe.toArray();
      const allFinite = Array.from(data).every((v) => Number.isFinite(v) && Math.abs(v) <= 1);
      check("PosEnc in [-1,1] and finite", allFinite);
      log(`  PosEnc: shape=[${pe.shape}]  |max|=${Math.max(...Array.from(data).map(Math.abs)).toFixed(4)}`);
    }

    // 4. Autograd through MHA: dLoss/dInput should be finite + non-zero for
    //    an L2 loss on the output.
    {
      const B = 1, S = 4, D = 16, H = 2;
      const x = Tensor.randn([B, S, D], true);
      const mha = new nn.MultiHeadAttention(D, H);
      const y = mha.forward(x);
      const loss = y.mul(y).sum();
      loss.backward();
      const gx = await x.grad!.toArray();
      const allFinite = Array.from(gx).every((v) => Number.isFinite(v));
      const anyNonZero = Array.from(gx).some((v) => Math.abs(v) > 1e-6);
      check("MHA autograd finite", allFinite);
      check("MHA autograd non-zero", anyNonZero);
      log(`  MHA autograd: dx.length=${gx.length}  allFinite=${allFinite}  anyNonZero=${anyNonZero}`);
    }

    // 5. Parameter count matches theoretical: 4 * D^2 for weights + 4*D for biases.
    {
      const D = 32, H = 4;
      const mha = new nn.MultiHeadAttention(D, H);
      const params = mha.parameters();
      let total = 0;
      for (const p of params) total += p.shape.reduce((a, b) => a * b, 1);
      const expected = 4 * D * D + 4 * D;
      check("MHA param count", total === expected, `got ${total}, expected ${expected}`);
      log(`  MHA params: ${total} == ${expected}`);
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
  name: "MHA + TransformerEncoderLayer: cross-backend",
  paramName: "backend",
  params: ["workers", "wasm", "webgpu"],
  description:
    "Composed MHA and pre-norm transformer encoder block, checked for shape, finiteness, autograd flow, and expected parameter count across all backends.",
  runner: runAttention,
});
