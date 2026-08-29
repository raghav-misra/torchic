import { Tensor, init, shutdown } from "../src/index";
import { parseSafetensors } from "../src/nn/safetensors";
import type { SafetensorsMap } from "../src/nn/safetensors";
import { Embedding, RMSNorm } from "../src/nn/layers";
import { KVCache } from "../src/nn/kv_cache";
import * as functional from "../src/nn/functional";
import { LLAMA_3_2_1B } from "./demos/llama/config";
import { LlamaDecoderLayer } from "./demos/llama/model";
import {
  parsePartialFromFile,
  isEmbedKey,
  isFinalNormKey,
  isLayerKey,
} from "./demos/llama/weights";

const status = document.getElementById("status")!;
const logEl = document.getElementById("log")!;
function log(msg: string): void {
  logEl.textContent += msg + "\n";
  console.log(`[bench] ${msg}`);
}

function waitForFile(inputId: string): Promise<File> {
  return new Promise((resolve, reject) => {
    const input = document.getElementById(inputId) as HTMLInputElement | null;
    if (!input) return reject(new Error(`missing input #${inputId}`));
    if (input.files && input.files.length > 0) {
      resolve(input.files[0]);
      return;
    }
    input.addEventListener(
      "change",
      () => {
        const f = input.files?.[0];
        if (f) resolve(f);
        else reject(new Error(`#${inputId}: no file`));
      },
      { once: true },
    );
  });
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

function cosineSim(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

function stripPrefix(sd: SafetensorsMap, prefix: string): SafetensorsMap {
  const out: SafetensorsMap = {};
  for (const [k, v] of Object.entries(sd)) {
    if (k.startsWith(prefix)) out[k.slice(prefix.length)] = v;
    else out[k] = v;
  }
  return out;
}

interface StageResult { stage: string; diff: number; cos: number; tol: number; ok: boolean }

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const backend = (params.get("backend") ?? "workers") as "webgpu" | "workers" | "wasm";
  const memorySizeMB = parseInt(params.get("memory") ?? "2000", 10);
  const layersParam = params.get("layers") ?? "0,15";
  const layerIndices = layersParam === "all"
    ? [...Array(LLAMA_3_2_1B.numLayers).keys()]
    : layersParam.split(",").map((s) => parseInt(s, 10)).filter((n) => Number.isFinite(n));

  log(`config: backend=${backend} memory=${memorySizeMB}MB layers=${layerIndices.join(",")}`);

  status.textContent = "init";
  await init({ backend, memorySizeMB });
  log(`backend ready`);

  status.textContent = "await-ref";
  log(`waiting for reference dump .safetensors (upload to #ref-input)...`);
  const refFile = await waitForFile("ref-input");
  log(`reading ${refFile.name} (${(refFile.size / 1e6).toFixed(1)} MB)`);
  const refBuf = await refFile.arrayBuffer();
  const ref = parseSafetensors(refBuf);
  const tokenIds = Array.from(ref["token_ids"].data);
  const T = tokenIds.length;
  log(`  T=${T}, tokens=[${tokenIds.slice(0, 12).join(",")}${T > 12 ? "..." : ""}]`);

  status.textContent = "await-model";
  log(`waiting for model .safetensors (upload to #model-input)...`);
  const modelFile = await waitForFile("model-input");
  log(`model: ${modelFile.name} (${(modelFile.size / 1e9).toFixed(2)} GB)`);
  log(`  streaming tensor ranges via File.slice — full ArrayBuffer never allocated`);

  const results: StageResult[] = [];

  status.textContent = "test-embed";
  const embed = new Embedding(LLAMA_3_2_1B.vocabSize, LLAMA_3_2_1B.hiddenSize);
  {
    log(`[embed] streaming model.embed_tokens.weight...`);
    const sd = await parsePartialFromFile(modelFile, isEmbedKey);
    embed.load_safetensors({ weight: sd["model.embed_tokens.weight"] }, { strict: true });
    const idsTensor = Tensor.fromData(tokenIds, [T]);
    const out = embed.forward(idsTensor);
    const outData = await out.toArray();
    const refData = ref["hidden_00"].data;
    const diff = maxAbsDiff(outData, refData);
    const cos = cosineSim(outData, refData);
    const tol = 1e-4;
    const ok = diff <= tol;
    results.push({ stage: "embed", diff, cos, tol, ok });
    log(`  diff=${diff.toExponential(3)} cos=${cos.toFixed(6)} tol=${tol.toExponential(1)} ${ok ? "OK" : "FAIL"}`);
    idsTensor.dispose();
    out.dispose();
  }

  for (const layerIdx of layerIndices) {
    status.textContent = `test-layer-${layerIdx}`;
    log(`[layer ${layerIdx}] streaming model.layers.${layerIdx}.*...`);
    const sd = stripPrefix(
      await parsePartialFromFile(modelFile, isLayerKey(layerIdx)),
      `model.layers.${layerIdx}.`,
    );
    const layer = new LlamaDecoderLayer(LLAMA_3_2_1B);
    layer.load_safetensors(sd, { strict: true });

    const { cos: ropeCos, sin: ropeSin } = functional.precomputeRope(
      T,
      LLAMA_3_2_1B.headDim,
      LLAMA_3_2_1B.ropeTheta,
    );
    const cache = new KVCache(1, T, LLAMA_3_2_1B.numKvHeads, LLAMA_3_2_1B.headDim);

    const inKey = `hidden_${String(layerIdx).padStart(2, "0")}`;
    // Layer i's output is hidden_{i+1} for i<N-1 (which is layer i+1's input,
    // pre-norm). For the final layer, its pre-norm output was captured via a
    // forward-pre-hook on model.norm and stored under `pre_final_norm`.
    const outKey =
      layerIdx === LLAMA_3_2_1B.numLayers - 1
        ? "pre_final_norm"
        : `hidden_${String(layerIdx + 1).padStart(2, "0")}`;
    const inputRef = ref[inKey].data;
    const outputRef = ref[outKey].data;
    const input = Tensor.fromData(Array.from(inputRef), [T, LLAMA_3_2_1B.hiddenSize]);
    const out = layer.forward(input, ropeCos, ropeSin, cache, 0, 0);
    const outData = await out.toArray();
    const diff = maxAbsDiff(outData, outputRef);
    const cos = cosineSim(outData, outputRef);
    // Residual streams grow with depth so absolute diff isn't a great yardstick;
    // cosine similarity is the primary signal.
    const tol = 1e-1;
    const ok = diff <= tol;
    results.push({ stage: `layer ${layerIdx}`, diff, cos, tol, ok });
    log(`  diff=${diff.toExponential(3)} cos=${cos.toFixed(6)} tol=${tol.toExponential(1)} ${ok ? "OK" : "FAIL"}`);

    input.dispose();
    out.dispose();
    cache.dispose();
    ropeCos.dispose();
    ropeSin.dispose();
  }

  status.textContent = "test-final-norm";
  {
    log(`[final norm] streaming model.norm.weight...`);
    const sd = await parsePartialFromFile(modelFile, isFinalNormKey);
    const norm = new RMSNorm(LLAMA_3_2_1B.hiddenSize, LLAMA_3_2_1B.rmsEps);
    norm.load_safetensors({ weight: sd["model.norm.weight"] }, { strict: true });
    // Input is layer 15's pre-norm output (captured via hook); expected is HF's
    // last hidden_states entry which is the post-final-norm state.
    const inputRef = ref["pre_final_norm"].data;
    const outputRef = ref[`hidden_${String(LLAMA_3_2_1B.numLayers).padStart(2, "0")}`].data;
    const input = Tensor.fromData(Array.from(inputRef), [T, LLAMA_3_2_1B.hiddenSize]);
    const out = norm.forward(input);
    const outData = await out.toArray();
    const diff = maxAbsDiff(outData, outputRef);
    const cos = cosineSim(outData, outputRef);
    const tol = 5e-3;
    const ok = diff <= tol;
    results.push({ stage: "final norm", diff, cos, tol, ok });
    log(`  diff=${diff.toExponential(3)} cos=${cos.toFixed(6)} tol=${tol.toExponential(1)} ${ok ? "OK" : "FAIL"}`);
    input.dispose();
    out.dispose();
  }

  status.textContent = "test-lm-head";
  {
    log(`[lm head] tied-embed matmul on hidden_${LLAMA_3_2_1B.numLayers}...`);
    const postNorm = Tensor.fromData(
      Array.from(ref[`hidden_${String(LLAMA_3_2_1B.numLayers).padStart(2, "0")}`].data),
      [T, LLAMA_3_2_1B.hiddenSize],
    );
    const logits = postNorm.matmul(embed.W.transpose(-1, -2));
    const outData = await logits.toArray();
    const refLogits = ref["logits"].data;
    const diff = maxAbsDiff(outData, refLogits);
    const cos = cosineSim(outData, refLogits);
    const tol = 5e-2;
    const ok = diff <= tol;
    results.push({ stage: "lm head", diff, cos, tol, ok });
    log(`  diff=${diff.toExponential(3)} cos=${cos.toFixed(6)} tol=${tol.toExponential(1)} ${ok ? "OK" : "FAIL"}`);

    // Also report top-5 predicted tokens vs reference top-5 at the last position.
    const lastRowStart = (T - 1) * LLAMA_3_2_1B.vocabSize;
    const ourRow = outData.subarray(lastRowStart, lastRowStart + LLAMA_3_2_1B.vocabSize);
    const refRow = refLogits.subarray(lastRowStart, lastRowStart + LLAMA_3_2_1B.vocabSize);
    const topK = (row: Float32Array, k: number) => {
      const idx = [...row.keys()].sort((a, b) => row[b] - row[a]).slice(0, k);
      return idx.map((i) => [i, row[i]] as [number, number]);
    };
    const ourTop = topK(ourRow, 5);
    const refTop = topK(refRow, 5);
    log(`  our top-5: ${ourTop.map(([i, v]) => `${i}(${v.toFixed(2)})`).join(", ")}`);
    log(`  ref top-5: ${refTop.map(([i, v]) => `${i}(${v.toFixed(2)})`).join(", ")}`);
    postNorm.dispose();
    logits.dispose();
  }

  shutdown();

  const passed = results.filter((r) => r.ok).length;
  log(``);
  log(`summary: ${passed}/${results.length} stages match`);
  for (const r of results) {
    log(`  ${r.stage.padEnd(14)} diff=${r.diff.toExponential(3)} cos=${r.cos.toFixed(6)} ${r.ok ? "OK" : "FAIL"}`);
  }

  status.textContent = "done";
  console.log(`__RESULT__${JSON.stringify({ passed, total: results.length, results })}`);
}

main().catch((e) => {
  const msg = e instanceof Error ? e.stack ?? e.message : String(e);
  log(`error: ${msg}`);
  console.log(`__ERROR__${msg}`);
  status.textContent = "error";
});
