import { Tensor, trackTensors, crossEntropy, noGrad } from "../src/index";

// --- Helpers ---------------------------------------------------------------
function buildVocab(chars: string) {
  const stoi: { [key: string]: number } = {};
  const itos: { [key: number]: string } = {};
  for (let i = 0; i < chars.length; i++) {
    stoi[chars[i]] = i;
    itos[i] = chars[i];
  }
  return { stoi, itos };
}

function buildDataset(
  names: string[],
  stoi: { [key: string]: number },
  blockSize: number
) {
  const Xarray: number[][] = [];
  const Yarray: number[] = [];

  for (const word of names) {
    const context = new Array(blockSize).fill(".");
    for (const char of word + ".") {
      const ix = stoi[char];
      Xarray.push(context.map((c) => stoi[c]));
      Yarray.push(ix);
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
  hiddenSize: number
) {
  // loose peak floats: params+grads + batch temps (embeddings, hidden, logits/softmax)
  const params =
    vocabSize * embeddingDims + // Wembed
    blockSize * embeddingDims * hiddenSize + // Whidden
    hiddenSize * vocabSize + // Wout
    hiddenSize +
    vocabSize; // biases

  const peakFloats =
    2 * params +
    batchSize * (blockSize * embeddingDims + hiddenSize + 2 * vocabSize);
  const estimatedMB = Math.ceil((peakFloats * 4) / (1024 * 1024)) + 8; // add slack
  return estimatedMB;
}

function detectCores(): number {
  return navigator?.hardwareConcurrency ?? 4;
}

function chooseThreadCount(
  rows: number,
  minRowsPerThread = 8,
  maxThreads = 8
): number {
  const cores = detectCores();
  const maxByWork = Math.max(
    1,
    Math.floor(rows / Math.max(1, minRowsPerThread))
  );
  return Math.min(cores, maxByWork, maxThreads);
}

function createModel(
  vocabSize: number,
  embeddingDims: number,
  blockSize: number,
  hiddenSize: number,
  initialLR: number
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

// helper to sample index from probs Float32Array
function sampleFromProbArray(arr: Float32Array) {
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i];
  let r = Math.random() * sum;
  for (let i = 0; i < arr.length; i++) {
    r -= arr[i];
    if (r <= 0) return i;
  }
  return arr.length - 1;
}

// --- Main -----------------------------------------------------------------
async function makemoreMLP() {
  const names = await fetch(
    "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
  )
    .then((res) => res.text())
    .then((text) => text.split("\n").filter((n) => n.length > 0));

  console.log("Number of names:", names.length);

  const chars = "abcdefghijklmnopqrstuvwxyz.";
  const { stoi, itos } = buildVocab(chars);

  // hyperparameters
  const vocabSize = chars.length; // 26 letters + '.'
  const embeddingDims = 10;
  const blockSize = 5;
  const batchSize = 512;
  const hiddenSize = 300;
  // early stopping threshold (stop when epoch avg loss <= threshold)
  const earlyStopThreshold = 2.3;

  const estimatedMB = estimateMemoryMB(
    vocabSize,
    embeddingDims,
    blockSize,
    batchSize,
    hiddenSize
  );
  console.log(`Estimated memory (MB): ${estimatedMB}`);

  const threads = chooseThreadCount(Math.max(1, batchSize));
  console.log(`Chosen threads: ${threads}`);

  const { Xarray, Yarray } = buildDataset(names, stoi, blockSize);

  // Initialize dispatcher with auto-chosen threads and estimated memory (MB)
  await Tensor.init(threads, estimatedMB);

  // Avoid creating full-dataset Tensors to reduce memory pressure.
  // Batches are created from the JS arrays (`Xarray`, `Yarray`) on-the-fly.
  console.log("Dataset rows (tokens):", Xarray.length);
  const datasetSize = Xarray.length;

  // learning rate schedule config
  const lrConfig = { initial: 0.1, decayRate: 0.95 };
  const numEpochs = 5; // run this many full passes over the dataset

  const { Wembed, Whidden, bhidden, Wout, bout, parameters, learningRate } =
    createModel(
      vocabSize,
      embeddingDims,
      blockSize,
      hiddenSize,
      lrConfig.initial
    );

  // JS-side current lr value (we update the scalar Tensor each schedule step)
  let lrValue = lrConfig.initial;

  // Training loop (minibatches) with shuffling per epoch
  const stepTimes: number[] = [];
  let windowSum = 0; // ms sum of last up to 100 steps

  // build indices and shuffle helper
  const indices = Array.from({ length: datasetSize }, (_, i) => i);
  function shuffleInPlace(arr: number[]) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
  }

  let globalStep = 0;

  // Preallocate reusable batch buffers to avoid per-batch allocations/copies
  const Xbuf = Tensor.empty([batchSize, blockSize]);
  const Ybuf = Tensor.empty([batchSize]);
  const flatX = new Float32Array(batchSize * blockSize);
  const flatY = new Float32Array(batchSize);

  for (let epoch = 0; epoch < numEpochs; epoch++) {
    shuffleInPlace(indices);
    let epochLossSum = 0;
    let epochStepCount = 0;

    for (let pos = 0; pos < datasetSize; pos += batchSize) {
      // decay learning rate on schedule (based on globalStep)
      const batchIdx = indices.slice(pos, pos + batchSize);
      const B = batchIdx.length;

      // fill flat buffers (reuse to avoid allocations)
      flatX.fill(0);
      for (let i = 0; i < B; i++) {
        const src = Xarray[batchIdx[i]];
        for (let j = 0; j < blockSize; j++) flatX[i * blockSize + j] = src[j];
      }
      flatY.fill(0);
      for (let i = 0; i < B; i++) flatY[i] = Yarray[batchIdx[i]];

      Xbuf.write(flatX);
      Ybuf.write(flatY);

      const Xbatch = Xbuf.slice([[0, B], [0, blockSize]]);
      const Ybatch = Ybuf.slice([[0, B]]);

      const t0 = performance.now();

      const lossValue = await trackTensors(async () => {
        for (const p of parameters) p.grad = null;

        const emb = Wembed.embedding(Xbatch); // [B, blockSize, embeddingDims]
        const B = emb.shape[0];
        const embFlat = emb.reshape([B, blockSize * embeddingDims]);

        const hidden = embFlat.matmul(Whidden).add(bhidden).tanh();
        const logits = hidden.matmul(Wout).add(bout);

        const loss = crossEntropy(logits, Ybatch);

        loss.backward();

        for (const p of parameters) p.sub_(p.grad!.mul(learningRate));

        return await loss.item();
      });

      const t1 = performance.now();
      const elapsedMs = t1 - t0;
      stepTimes.push(elapsedMs);
      windowSum += elapsedMs;
      if (stepTimes.length > 100)
        windowSum -= stepTimes[stepTimes.length - 101];

      epochLossSum += lossValue;
      epochStepCount++;

      if (globalStep % 100 === 0) {
        console.log(
          `Step ${globalStep}, loss: ${lossValue} (step time ${(
            elapsedMs / 1000
          ).toFixed(3)}s)`
        );
      }

      if ((globalStep + 1) % 100 === 0) {
        const count = Math.min(100, stepTimes.length);
        const avgMs = windowSum / count;
        console.log(
          `Avg step time (last ${count}): ${(avgMs / 1000).toFixed(3)}s`
        );
      }

      globalStep++;
    }

    // end of epoch: report epoch stats and decay LR
    const epochAvgLoss = epochStepCount > 0 ? epochLossSum / epochStepCount : 0;
    console.log(
      `Epoch ${
        epoch + 1
      }/${numEpochs} finished — avg loss: ${epochAvgLoss.toFixed(
        6
      )} — steps this epoch: ${epochStepCount}`
    );

    // decay learning rate once per epoch
    lrValue *= lrConfig.decayRate;
    learningRate.set([0], lrValue);
    console.log(`Decayed learning rate after epoch ${epoch + 1}: ${lrValue}`);

    // early stopping check
    if (epochAvgLoss <= earlyStopThreshold) {
      console.log(
        `Early stopping: epochAvgLoss=${epochAvgLoss.toFixed(
          6
        )} <= ${earlyStopThreshold}`
      );
      break;
    }
  }

  // --- Inference: generate 10 names starting from context "..." ---
  console.log("Starting inference (10 samples)");

  await noGrad(async () => {
    for (let sampleIdx = 0; sampleIdx < 10; sampleIdx++) {
      let context = new Array(blockSize).fill(".");
      let generated = "";

      while (true) {
        // build input tensor and run forward inside trackTensors so temps get disposed
        const probsArr: Float32Array = await trackTensors(async () => {
          const Xctx = Tensor.fromData([context.map((c) => stoi[c])]); // shape [1, B]
          const emb = Wembed.embedding(Xctx);
          const embFlat = emb.reshape([1, blockSize * embeddingDims]);
          const hidden = embFlat.matmul(Whidden).add(bhidden).relu();
          const logits = hidden.matmul(Wout).add(bout);
          const probs = logits.softmax(-1);
          return await probs.toArray();
        });

        const ix = sampleFromProbArray(probsArr);
        const ch = itos[ix];
        if (ch === ".") break;
        generated += ch;

        // advance context
        context.shift();
        context.push(ch);
      }

      console.log(`Sample ${sampleIdx + 1}: ${generated}`);
    }
  });
}

makemoreMLP();
