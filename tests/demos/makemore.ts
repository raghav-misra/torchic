import { Tensor, trackTensors, crossEntropy, noGrad, init, shutdown, nn, optim } from "../../src/index";
import { defineBench } from "../framework/define";
import type { BenchMetrics, RunContext } from "../framework/types";

type Backend = "workers" | "wasm" | "webgpu";

const NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt";
const CHARS = "abcdefghijklmnopqrstuvwxyz.";

const EMBEDDING_DIMS = 10;
const BLOCK_SIZE = 3;
const BATCH_SIZE = 256;
const HIDDEN_SIZE = 200;
const INITIAL_LR = 0.1;
const WARMUP_STEPS = 10;
const MEASURED_STEPS = 300;
const NUM_SAMPLES = 5;

// Karpathy's makemore MLP: char embedding -> flatten -> tanh hidden -> logits.
class MakemoreMLP extends nn.Module {
  emb: nn.Embedding;
  hidden: nn.Linear;
  out: nn.Linear;

  constructor(vocab: number, embDim: number, block: number, hidden: number) {
    super();
    this.emb = this.child("emb", new nn.Embedding(vocab, embDim));
    this.hidden = this.child("hidden", new nn.Linear(block * embDim, hidden));
    this.out = this.child("out", new nn.Linear(hidden, vocab));
  }

  forward(x: Tensor): Tensor {
    const [B] = x.shape;
    const e = this.emb.forward(x).reshape([B, -1]);
    const h = this.hidden.forward(e).tanh();
    return this.out.forward(h);
  }
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return NaN;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))));
  return sorted[idx];
}

let cachedNames: string[] | null = null;
async function loadNames() {
  if (cachedNames) return cachedNames;
  const text = await fetch(NAMES_URL).then((r) => r.text());
  cachedNames = text.split("\n").filter((n) => n.length > 0);
  return cachedNames;
}

function buildVocab(chars: string) {
  const stoi: Record<string, number> = {};
  const itos: Record<number, string> = {};
  for (let i = 0; i < chars.length; i++) {
    stoi[chars[i]] = i;
    itos[i] = chars[i];
  }
  return { stoi, itos };
}

function buildDataset(names: string[], stoi: Record<string, number>, blockSize: number) {
  const Xarray: number[][] = [];
  const Yarray: number[] = [];
  for (const word of names) {
    const context = new Array(blockSize).fill(".");
    for (const char of word + ".") {
      Xarray.push(context.map((c) => stoi[c]));
      Yarray.push(stoi[char]);
      context.shift();
      context.push(char);
    }
  }
  return { Xarray, Yarray };
}

// Params + grads + 2x fudge on peak activations to cover autograd retention.
function estimateMemoryMB(vocabSize: number) {
  const params =
    vocabSize * EMBEDDING_DIMS +
    BLOCK_SIZE * EMBEDDING_DIMS * HIDDEN_SIZE +
    HIDDEN_SIZE * vocabSize +
    HIDDEN_SIZE +
    vocabSize;
  const activations =
    BATCH_SIZE * (BLOCK_SIZE * EMBEDDING_DIMS + HIDDEN_SIZE + 2 * vocabSize);
  const peakFloats = 2 * params + 2 * activations;
  return Math.ceil((peakFloats * 4) / (1024 * 1024)) + 16;
}

function sampleFromProbArray(arr: Float32Array) {
  let sum = 0;
  for (const val of arr) sum += val;
  let r = Math.random() * sum;
  for (let i = 0; i < arr.length; i++) {
    r -= arr[i];
    if (r <= 0) return i;
  }
  return arr.length - 1;
}

function shuffleInPlace(arr: number[]) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

export async function runMakemore(
  backend: Backend,
  { log }: RunContext,
  threadsOverride?: number,
): Promise<BenchMetrics> {
  const { stoi, itos } = buildVocab(CHARS);
  const vocabSize = CHARS.length;

  log("fetching names.txt...");
  const names = await loadNames();
  log(`  ${names.length} names`);

  const { Xarray, Yarray } = buildDataset(names, stoi, BLOCK_SIZE);
  log(`  ${Xarray.length} training rows`);

  const estimatedMB = estimateMemoryMB(vocabSize);
  const threads =
    threadsOverride ??
    (backend === "webgpu" ? 1 : Math.min(navigator.hardwareConcurrency ?? 4, 8));
  log(`init ${backend} (threads=${threads}, ${estimatedMB} MB)`);
  await init({ backend, threadCount: threads, memorySizeMB: estimatedMB });

  try {
    const model = new MakemoreMLP(vocabSize, EMBEDDING_DIMS, BLOCK_SIZE, HIDDEN_SIZE);
    const opt = new optim.SGD(model.parameters(), INITIAL_LR);

    const Xbuf = Tensor.empty([BATCH_SIZE, BLOCK_SIZE]);
    const Ybuf = Tensor.empty([BATCH_SIZE]);
    const flatX = new Float32Array(BATCH_SIZE * BLOCK_SIZE);
    const flatY = new Float32Array(BATCH_SIZE);

    const indices = Array.from({ length: Xarray.length }, (_, i) => i);
    shuffleInPlace(indices);

    const stepTimes: number[] = [];
    let lastLoss = Number.POSITIVE_INFINITY;
    let dataPos = 0;
    let epoch = 0;

    const totalSteps = WARMUP_STEPS + MEASURED_STEPS;
    log(`training ${WARMUP_STEPS} warmup + ${MEASURED_STEPS} measured steps @ batch=${BATCH_SIZE}...`);
    for (let step = 0; step < totalSteps; step++) {
      if (dataPos + BATCH_SIZE > indices.length) {
        shuffleInPlace(indices);
        dataPos = 0;
        epoch++;
      }
      const batchIdx = indices.slice(dataPos, dataPos + BATCH_SIZE);
      dataPos += BATCH_SIZE;

      for (let i = 0; i < BATCH_SIZE; i++) {
        const src = Xarray[batchIdx[i]];
        for (let j = 0; j < BLOCK_SIZE; j++) flatX[i * BLOCK_SIZE + j] = src[j];
        flatY[i] = Yarray[batchIdx[i]];
      }
      Xbuf.write(flatX);
      Ybuf.write(flatY);

      const t0 = performance.now();
      lastLoss = await trackTensors(async () => {
        opt.zeroGrad();
        const loss = crossEntropy(model.forward(Xbuf), Ybuf);
        loss.backward();
        await opt.step();
        return await loss.item();
      });
      const dt = performance.now() - t0;
      if (step >= WARMUP_STEPS) stepTimes.push(dt);

      if (step % 50 === 0 || step === WARMUP_STEPS) {
        const tag = step < WARMUP_STEPS ? "warmup" : "measured";
        log(`  ${tag} step ${step}  loss=${lastLoss.toFixed(4)}  ep=${epoch}  (${dt.toFixed(1)}ms)`);
      }
    }

    const sorted = [...stepTimes].sort((a, b) => a - b);
    const totalMs = stepTimes.reduce((a, b) => a + b, 0);
    const meanMs = totalMs / stepTimes.length;
    const medianMs = percentile(sorted, 0.5);
    const p95Ms = percentile(sorted, 0.95);
    const stepsPerSec = 1000 / meanMs;
    const samplesPerSec = stepsPerSec * BATCH_SIZE;

    log(`sampling ${NUM_SAMPLES} names (not measured)...`);
    const samples: string[] = [];
    await noGrad(async () => {
      let attempts = 0;
      while (samples.length < NUM_SAMPLES && attempts < NUM_SAMPLES * 4) {
        attempts++;
        const context = new Array(BLOCK_SIZE).fill(".");
        let generated = "";
        while (generated.length < 20) {
          const probs: Float32Array = await trackTensors(async () => {
            const Xctx = Tensor.fromData([context.map((c) => stoi[c])]);
            const logits = model.forward(Xctx);
            return await logits.softmax(-1).toArray();
          });
          const ix = sampleFromProbArray(probs);
          const ch = itos[ix];
          if (ch === ".") break;
          generated += ch;
          context.shift();
          context.push(ch);
        }
        if (generated.length === 0) continue;
        samples.push(generated);
        log(`  ${generated}`);
      }
    });

    return {
      "mean ms/step": Number(meanMs.toFixed(2)),
      "median ms/step": Number(medianMs.toFixed(2)),
      "p95 ms/step": Number(p95Ms.toFixed(2)),
      "steps/s": Number(stepsPerSec.toFixed(1)),
      "samples/s": Number(samplesPerSec.toFixed(0)),
      "training total (s)": Number((totalMs / 1000).toFixed(2)),
      "final loss": Number(lastLoss.toFixed(4)),
      samples: samples.join(", "),
    };
  } finally {
    shutdown();
  }
}

defineBench<Backend>({
  name: "Makemore MLP: training step latency",
  paramName: "backend",
  params: ["workers", "wasm", "webgpu"],
  description:
    "Karpathy's char-level MLP built with `nn.Module` + `optim.SGD` (27-token vocab, 3-char context, 10-D embedding, 200-unit hidden). Discards the first 10 warmup steps, then times 300 SGD training steps at batch 256 on the chosen backend. Reports per-step latency (mean/median/p95), throughput (steps/s, samples/s), and total training wall time. Fetch, dataset build, model init, and sampling are logged but excluded from the reported latency.",
  highlight: ["median ms/step", "samples/s"],
  runner: runMakemore,
});
