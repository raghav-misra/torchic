import { Tensor, init, shutdown, nn } from "../../src/index";
import { defineTest } from "../framework/define";
import type { RunContext, TestResult } from "../framework/types";
import { parseSafetensors, saveSafetensors } from "../../src/nn/safetensors";

type Backend = "workers" | "wasm" | "webgpu";

class TinyMLP extends nn.Module {
  emb: nn.Embedding;
  hidden: nn.Linear;
  out: nn.Linear;

  constructor(vocab: number, embDim: number, hidden: number) {
    super();
    this.emb = this.child("emb", new nn.Embedding(vocab, embDim));
    this.hidden = this.child("hidden", new nn.Linear(embDim, hidden));
    this.out = this.child("out", new nn.Linear(hidden, vocab));
  }

  forward(x: Tensor): Tensor {
    const h = this.hidden.forward(this.emb.forward(x)).tanh();
    return this.out.forward(h);
  }
}

async function runRoundtrip(backend: Backend, { log }: RunContext): Promise<TestResult> {
  const threadCount = backend === "webgpu" ? 1 : 4;
  await init({ backend, threadCount, memorySizeMB: 32 });
  try {
    const src = new TinyMLP(10, 4, 8);

    // Snapshot every parameter's data.
    const before: Record<string, Float32Array> = {};
    const sd = src.state_dict();
    for (const key of Object.keys(sd)) {
      before[key] = await sd[key].toArray();
    }

    // Save -> parse.
    const saveMap: Record<string, { shape: number[]; data: Float32Array }> = {};
    for (const key of Object.keys(sd)) {
      saveMap[key] = { shape: sd[key].shape, data: before[key] };
    }
    const buf = saveSafetensors(saveMap);
    const parsed = parseSafetensors(buf);

    // Load into a fresh module (fresh random init) and check equality.
    const dst = new TinyMLP(10, 4, 8);
    const dstBefore: Record<string, Float32Array> = {};
    const dstSd = dst.state_dict();
    for (const key of Object.keys(dstSd)) {
      dstBefore[key] = await dstSd[key].toArray();
    }
    // Sanity: fresh init should differ from src.
    let diffCount = 0;
    for (const key of Object.keys(before)) {
      const a = before[key];
      const b = dstBefore[key];
      let any = false;
      for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > 1e-6) { any = true; break; }
      if (any) diffCount++;
    }
    log(`  ${diffCount}/${Object.keys(before).length} params differ between fresh + src (expected all)`);

    dst.load_safetensors(parsed);

    let maxErr = 0;
    let numChecked = 0;
    for (const key of Object.keys(before)) {
      const got = await dstSd[key].toArray();
      const want = before[key];
      for (let i = 0; i < got.length; i++) {
        const d = Math.abs(got[i] - want[i]);
        if (d > maxErr) maxErr = d;
      }
      numChecked++;
    }
    log(`  round-trip ${numChecked} params, maxErr=${maxErr.toExponential(2)}`);

    // Verify forward outputs match to numerical precision.
    const x = Tensor.fromData([0, 3, 7, 5, 1], [1, 5]);
    const y1 = await src.forward(x.reshape([-1])).toArray();
    const y2 = await dst.forward(x.reshape([-1])).toArray();
    let maxOutErr = 0;
    for (let i = 0; i < y1.length; i++) {
      const d = Math.abs(y1[i] - y2[i]);
      if (d > maxOutErr) maxOutErr = d;
    }
    log(`  forward output maxErr=${maxOutErr.toExponential(2)}`);

    // Strict mode should reject a missing key.
    let strictRejected = false;
    try {
      const bad = { ...parsed };
      delete bad["hidden.weight"];
      const dst2 = new TinyMLP(10, 4, 8);
      dst2.load_safetensors(bad, { strict: true });
    } catch {
      strictRejected = true;
    }
    log(`  strict rejected missing key: ${strictRejected}`);

    // renameMap should translate keys.
    const renameMap: Record<string, string> = {};
    const renamed: Record<string, typeof parsed[string]> = {};
    for (const key of Object.keys(parsed)) {
      const alt = "prefix." + key;
      renamed[alt] = parsed[key];
      renameMap[key] = alt;
    }
    const dst3 = new TinyMLP(10, 4, 8);
    dst3.load_safetensors(renamed, { renameMap });
    let renamedOk = true;
    const dst3Sd = dst3.state_dict();
    for (const key of Object.keys(before)) {
      const got = await dst3Sd[key].toArray();
      for (let i = 0; i < got.length; i++) if (Math.abs(got[i] - before[key][i]) > 1e-6) { renamedOk = false; break; }
      if (!renamedOk) break;
    }
    log(`  rename-map load: ${renamedOk ? "ok" : "FAIL"}`);

    // weight_norm fuse-on-load: build a synthetic (weight_g, weight_v) pair
    // and check that `load_safetensors` reconstructs `weight = g * v / ||v||`.
    let weightNormOk = true;
    let weightNormErr = 0;
    {
      class TinyConv extends nn.Module {
        conv: nn.Conv1d;
        constructor() { super(); this.conv = this.child("conv", new nn.Conv1d(2, 3, 5, { stride: 1, padding: 2 })); }
        forward(x: Tensor): Tensor { return this.conv.forward(x); }
      }
      const m = new TinyConv();
      const Cout = 3, Cin = 2, K = 5;
      const vData = new Float32Array(Cout * Cin * K);
      for (let i = 0; i < vData.length; i++) vData[i] = Math.sin(i * 0.31) + 0.1;
      const gData = new Float32Array(Cout);
      for (let o = 0; o < Cout; o++) gData[o] = 0.5 + o * 0.7;
      const biasData = new Float32Array(Cout);

      const wnSd = {
        "conv.weight_g": { shape: [Cout, 1, 1], data: gData },
        "conv.weight_v": { shape: [Cout, Cin, K], data: vData },
        "conv.bias": { shape: [Cout], data: biasData },
      };
      m.load_safetensors(wnSd);
      const loaded = await m.conv.weight.toArray();

      const expected = new Float32Array(Cout * Cin * K);
      for (let o = 0; o < Cout; o++) {
        let sq = 0;
        for (let i = 0; i < Cin * K; i++) sq += vData[o * Cin * K + i] ** 2;
        const norm = Math.sqrt(sq);
        for (let i = 0; i < Cin * K; i++) expected[o * Cin * K + i] = vData[o * Cin * K + i] * gData[o] / norm;
      }
      for (let i = 0; i < loaded.length; i++) {
        const d = Math.abs(loaded[i] - expected[i]);
        if (d > weightNormErr) weightNormErr = d;
      }
      weightNormOk = weightNormErr < 1e-5;
    }
    log(`  weight_norm fuse-on-load: ${weightNormOk ? "ok" : "FAIL"} (maxErr=${weightNormErr.toExponential(2)})`);

    const allOk = maxErr < 1e-6 && maxOutErr < 1e-4 && strictRejected && renamedOk && weightNormOk;
    return {
      pass: allOk,
      message: allOk ? `round-trip ok (params ${maxErr.toExponential(1)}, fwd ${maxOutErr.toExponential(1)})` : "some checks failed",
    };
  } finally {
    shutdown();
  }
}

defineTest<Backend>({
  name: "safetensors: round-trip Module save/load across backends",
  paramName: "backend",
  params: ["workers", "wasm", "webgpu"],
  description:
    "Save a TinyMLP's state_dict as safetensors, parse, load into a fresh model, verify parameter and forward-output equality. Also exercises strict-mode rejection and renameMap.",
  runner: runRoundtrip,
});
