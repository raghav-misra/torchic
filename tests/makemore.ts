import { Tensor, trackTensors, crossEntropy } from "../src/index";

// one hot helper using tensors
function oneHot(index: number, dims: number): Tensor {
  const tensor = Tensor.zeros([dims]);
  tensor.setValue([index], 1);
  return tensor;
}

async function makemoreMLP() {
  await Tensor.init(8, 1024);

  /*
    3 Character block size.
    Input is transformed into embeddings.
    Embeddings transformed by hidden layer of weights and biases.
    Then character probabilities are yielded with softmax.
  */

  const response = await fetch(
    "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
  );
  const text = await response.text();
  const names = text.split("\n").filter((n) => n.length > 0);

  const blockSize = 3;
  const Xarray: number[] = [];
  const Yarray: number[] = [];

  // init maps for all letters to integers and back
  const stoi: { [key: string]: number } = {};
  const itos: { [key: number]: string } = {};
  const chars = "abcdefghijklmnopqrstuvwxyz.";
  for (let i = 0; i < chars.length; i++) {
    stoi[chars[i]] = i;
    itos[i] = chars[i];
  }

  // build the dataset
  for (const word of names) {
    const context = new Array(3).fill(".");

    for (const char of word + ".") {
      const ix = stoi[char];

      for (const c of context) {
        Xarray.push(stoi[c]);
      }
      Yarray.push(ix);

      context.shift();
      context.push(char);
    }
  }

  // convert arrays to tensors
  // We will use minibatches instead of full dataset
  // const X = Tensor.fromData(Xarray, [Xarray.length / blockSize, blockSize]);
  // const Y = Tensor.fromData(Yarray, [Yarray.length, 1]);

  // Model hyperparameters
  const embeddingDim = 24;
  const hiddenSize = 500;
  const scale = Tensor.fromData([0.1], [1]);

  const C = Tensor.randn([27, embeddingDim], true).mul_(scale);
  const W1 = Tensor.randn([blockSize * embeddingDim, hiddenSize], true).mul_(
    scale
  );
  const b1 = Tensor.zeros([hiddenSize], true);
  const W2 = Tensor.randn([hiddenSize, 27], true).mul_(scale);
  const b2 = Tensor.zeros([27], true);

  const parameters = [C, W1, b1, W2, b2];

  const steps = 30000;
  const batchSize = 256;

  console.log(`Starting training for ${steps} steps...`);

  for (let i = 0; i < steps; i++) {
    await trackTensors(async () => {
      // Construct minibatch
      const batchX: number[] = [];
      const batchY: number[] = [];

      for (let j = 0; j < batchSize; j++) {
        const idx = Math.floor(Math.random() * Yarray.length);
        batchX.push(Xarray[idx * 3], Xarray[idx * 3 + 1], Xarray[idx * 3 + 2]);
        batchY.push(Yarray[idx]);
      }

      const Xb = Tensor.fromData(batchX, [batchSize, blockSize]);

      // Loss target
      const Y_onehot_array = new Float32Array(batchSize * 27);
      for (let j = 0; j < batchSize; j++) {
        Y_onehot_array[j * 27 + batchY[j]] = 1.0;
      }
      const Y_onehot = Tensor.fromData(Y_onehot_array, [batchSize, 27]);

      // Forward
      const embeddings = C.embedding(Xb);
      const embCat = embeddings.reshape([batchSize, blockSize * embeddingDim]);
      const h = embCat.matmul(W1).add(b1).relu();
      const logits = h.matmul(W2).add(b2);

      const loss = crossEntropy(logits, Y_onehot);

      if (i % 100 === 0) {
        const currentLoss = await loss.item();
        console.log(`Step ${i}, Loss: ${currentLoss}`);
        if (Number.isNaN(currentLoss)) {
          console.log("Loss is NaN, stopping training");
          throw new Error("Loss is NaN, stopping training");
        }
      }

      // Zero Grads
      for (const p of parameters) {
        p.grad = null;
      }

      // Backward
      loss.backward();

      // Update
      const currentLr = i < 20000 ? 0.1 : 0.01;
      const lr = Tensor.fromData([currentLr], [1]);
      for (const p of parameters) {
        const grad = p.grad as Tensor | null;
        if (grad) p.sub_(grad.mul(lr));
      }
    });

    // Early exit if we hit NaN
    if (i % 100 === 0 && Number.isNaN(await C.getValue([0, 0]))) {
      break;
    }
  }

  console.log("Training complete. Generating names...");

  // Inference
  function sample(probs: Float32Array): number {
    const r = Math.random();
    let cdf = 0;
    for (let i = 0; i < probs.length; i++) {
      cdf += probs[i];
      if (r < cdf) return i;
    }
    return probs.length - 1;
  }

  for (let i = 0; i < 10; i++) {
    let out = "";
    let contextIdx = [stoi["."], stoi["."], stoi["."]];

    while (true) {
      const ix = await trackTensors(async () => {
        const x = Tensor.fromData(contextIdx, [1, 3]);
        const emb = C.embedding(x);
        const embCat = emb.reshape([1, blockSize * embeddingDim]);
        const h = embCat.matmul(W1).add(b1).relu();
        const logits = h.matmul(W2).add(b2);
        const probs = logits.softmax();

        const probsArray = await probs.toArray();
        return sample(probsArray);
      });

      const char = itos[ix];
      out += char;
      if (char === ".") break;

      contextIdx.shift();
      contextIdx.push(ix);
    }
    console.log(out);
  }
}

makemoreMLP();
