import { Tensor, trackTensors, crossEntropy, noGrad, init } from "../src/index";

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

function estimateMemoryMB(
  vocabSize: number,
  embeddingDims: number,
  blockSize: number,
  batchSize: number,
  hiddenSize: number,
) {
  const params =
    vocabSize * embeddingDims +
    blockSize * embeddingDims * hiddenSize +
    hiddenSize * vocabSize +
    hiddenSize +
    vocabSize;

  const peakFloats =
    2 * params + batchSize * (blockSize * embeddingDims + hiddenSize + 2 * vocabSize);
  return Math.ceil((peakFloats * 4) / (1024 * 1024)) + 8;
}

function chooseThreadCount(rows: number, minRowsPerThread = 8, maxThreads = 8) {
  const cores = navigator?.hardwareConcurrency ?? 4;
  const maxByWork = Math.max(1, Math.floor(rows / Math.max(1, minRowsPerThread)));
  return Math.min(cores, maxByWork, maxThreads);
}

function createModel(
  vocabSize: number,
  embeddingDims: number,
  blockSize: number,
  hiddenSize: number,
  initialLR: number,
) {
  const Wembed = Tensor.randn([vocabSize, embeddingDims], true);
  const Whidden = Tensor.randn([blockSize * embeddingDims, hiddenSize], true);
  const bhidden = Tensor.zeros([1, hiddenSize], true);
  const Wout = Tensor.randn([hiddenSize, vocabSize], true);
  const bout = Tensor.zeros([1, vocabSize], true);

  const parameters: Tensor[] = [Wembed, Whidden, bhidden, Wout, bout];

  const learningRate = Tensor.fromData([initialLR]);
  const initScale = Tensor.fromData([0.01]);
  Wembed.mul_(initScale);
  Whidden.mul_(initScale);
  Wout.mul_(initScale);

  return { Wembed, Whidden, bhidden, Wout, bout, parameters, learningRate };
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

enum TrainState {
  Running,
  Paused,
  Stopped,
}
let trainState = TrainState.Running;
let resumeResolve: (() => void) | null = null;

function waitForResume(): Promise<void> {
  return new Promise((resolve) => {
    resumeResolve = resolve;
  });
}

// Opaque check so TS control-flow can't narrow across await boundaries
function isStopped() {
  return trainState === TrainState.Stopped;
}

function pause() {
  if (trainState !== TrainState.Running) return console.log("Can't pause — not running.");
  trainState = TrainState.Paused;
  console.log("⏸ Paused. Use __makemore.resume() / __makemore.sample() in the console.");
}

function resume() {
  if (trainState !== TrainState.Paused) return console.log("Can't resume — not paused.");
  trainState = TrainState.Running;
  resumeResolve?.();
  resumeResolve = null;
  console.log("▶ Training resumed.");
}

function stop() {
  if (trainState === TrainState.Stopped) return console.log("Already stopped.");
  const wasPaused = trainState === TrainState.Paused;
  trainState = TrainState.Stopped;
  if (wasPaused) {
    resumeResolve?.();
    resumeResolve = null;
  }
  console.log("⏹ Training stopped.");
}

function shuffleInPlace(arr: number[]) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

async function makemoreMLP() {
  const names = await fetch("https://raw.githubusercontent.com/karpathy/makemore/master/names.txt")
    .then((res) => res.text())
    .then((text) => text.split("\n").filter((n) => n.length > 0));

  console.log("Number of names:", names.length);

  const chars = "abcdefghijklmnopqrstuvwxyz.";
  const { stoi, itos } = buildVocab(chars);

  const vocabSize = chars.length;
  const embeddingDims = 10;
  const blockSize = 5;
  const batchSize = 512;
  const hiddenSize = 300;
  const earlyStopThreshold = 2.3;
  const numEpochs = 5;
  const lrDecayRate = 0.95;
  const initialLR = 0.1;

  const estimatedMB = estimateMemoryMB(vocabSize, embeddingDims, blockSize, batchSize, hiddenSize);
  const threads = chooseThreadCount(batchSize);
  console.log(`Memory: ${estimatedMB} MB, threads: ${threads}`);

  const { Xarray, Yarray } = buildDataset(names, stoi, blockSize);
  await init({ backend: "workers", threadCount: threads, memorySizeMB: estimatedMB });

  const datasetSize = Xarray.length;
  console.log("Dataset rows:", datasetSize);

  const { Wembed, Whidden, bhidden, Wout, bout, parameters, learningRate } = createModel(
    vocabSize,
    embeddingDims,
    blockSize,
    hiddenSize,
    initialLR,
  );

  async function sampleNames(count = 10) {
    await noGrad(async () => {
      for (let i = 0; i < count; i++) {
        const context = new Array(blockSize).fill(".");
        let generated = "";

        while (true) {
          const probs: Float32Array = await trackTensors(async () => {
            const Xctx = Tensor.fromData([context.map((c) => stoi[c])]);
            const emb = Wembed.embedding(Xctx);
            const embFlat = emb.reshape([1, blockSize * embeddingDims]);
            const hidden = embFlat.matmul(Whidden).add(bhidden).tanh();
            const logits = hidden.matmul(Wout).add(bout);
            return await logits.softmax(-1).toArray();
          });

          const ix = sampleFromProbArray(probs);
          const ch = itos[ix];
          if (ch === ".") break;
          generated += ch;
          context.shift();
          context.push(ch);
        }

        console.log(`Sample ${i + 1}: ${generated}`);
      }
    });
  }

  // @ts-expect-error Expose sampleNames for console access
  window.__makemore = { pause, resume, stop, sample: (n = 10) => sampleNames(n) };

  let lrValue = initialLR;
  const stepTimes: number[] = [];
  let windowSum = 0;
  const indices = Array.from({ length: datasetSize }, (_, i) => i);
  let globalStep = 0;

  const Xbuf = Tensor.empty([batchSize, blockSize]);
  const Ybuf = Tensor.empty([batchSize]);
  const flatX = new Float32Array(batchSize * blockSize);
  const flatY = new Float32Array(batchSize);

  for (let epoch = 0; epoch < numEpochs; epoch++) {
    if (trainState === TrainState.Stopped) break;
    shuffleInPlace(indices);
    let epochLossSum = 0;
    let epochStepCount = 0;

    for (let pos = 0; pos < datasetSize; pos += batchSize) {
      if (isStopped()) break;
      if (trainState === TrainState.Paused) await waitForResume();
      if (isStopped()) break;

      const batchIdx = indices.slice(pos, pos + batchSize);
      const B = batchIdx.length;

      flatX.fill(0);
      for (let i = 0; i < B; i++) {
        const src = Xarray[batchIdx[i]];
        for (let j = 0; j < blockSize; j++) flatX[i * blockSize + j] = src[j];
      }
      flatY.fill(0);
      for (let i = 0; i < B; i++) flatY[i] = Yarray[batchIdx[i]];

      Xbuf.write(flatX);
      Ybuf.write(flatY);

      const Xbatch = Xbuf.slice([
        [0, B],
        [0, blockSize],
      ]);
      const Ybatch = Ybuf.slice([[0, B]]);

      const t0 = performance.now();

      const lossValue = await trackTensors(async () => {
        for (const p of parameters) p.grad = null;

        const emb = Wembed.embedding(Xbatch);
        const embFlat = emb.reshape([B, blockSize * embeddingDims]);
        const hidden = embFlat.matmul(Whidden).add(bhidden).tanh();
        const logits = hidden.matmul(Wout).add(bout);
        const loss = crossEntropy(logits, Ybatch);

        loss.backward();
        for (const p of parameters) p.sub_(p.grad!.mul(learningRate));

        return await loss.item();
      });

      const elapsedMs = performance.now() - t0;
      stepTimes.push(elapsedMs);
      windowSum += elapsedMs;
      if (stepTimes.length > 100) windowSum -= stepTimes[stepTimes.length - 101];

      epochLossSum += lossValue;
      epochStepCount++;

      if (globalStep % 100 === 0) {
        console.log(`Step ${globalStep}, loss: ${lossValue} (${(elapsedMs / 1000).toFixed(3)}s)`);
      }

      if ((globalStep + 1) % 100 === 0) {
        const count = Math.min(100, stepTimes.length);
        console.log(`Avg step time (last ${count}): ${(windowSum / count / 1000).toFixed(3)}s`);
      }

      globalStep++;
    }

    const epochAvgLoss = epochStepCount > 0 ? epochLossSum / epochStepCount : 0;
    console.log(
      `Epoch ${epoch + 1}/${numEpochs} — avg loss: ${epochAvgLoss.toFixed(6)} — ${epochStepCount} steps`,
    );

    lrValue *= lrDecayRate;
    learningRate.set([0], lrValue);
    console.log(`LR after epoch ${epoch + 1}: ${lrValue}`);

    if (epochAvgLoss <= earlyStopThreshold) {
      console.log(`Early stop: loss ${epochAvgLoss.toFixed(6)} <= ${earlyStopThreshold}`);
      break;
    }
  }

  if (trainState === TrainState.Stopped) {
    console.log("Training was stopped early.");
  } else {
    console.log("Training complete. Generating 10 samples...");
    await sampleNames(10);
  }

  console.log("Controls: window.__makemore.{pause, resume, stop, sample}");
}

makemoreMLP();
