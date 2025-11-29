/**
 * Computes cross-entropy loss between logits and one-hot targets.
 * @param input Logits tensor [Batch, Classes]
 * @param target One-hot tensor [Batch, Classes]
 * @returns Scalar loss tensor
 */
export function crossEntropy(input: Tensor, target: Tensor): Tensor {
  // Accept either one-hot targets [N, C] or integer label tensor [N] (or [N,1]).
  const probs = input.softmax(-1);
  // Add epsilon to avoid log(0)
  const epsilon = Tensor.fromData([1e-7], [1]);
  const logProbs = probs.add(epsilon).log();

  // If target is integer labels (shape [N] or [N,1]), convert to one-hot using
  // an identity matrix and the existing `embedding` op to avoid any async reads.
  let targetsOneHot: Tensor;
  if (
    target.shape.length === 1 ||
    (target.shape.length === 2 && target.shape[1] === 1)
  ) {
    const N = target.shape[0];
    const C = input.shape[1];

    // Build identity matrix [C, C] as Float32Array (diagonal ones)
    const idData = new Float32Array(C * C);
    for (let i = 0; i < C; i++) idData[i * C + i] = 1;

    const identity = Tensor.fromData(idData, [C, C]);
    // embedding will produce shape [...target.shape, C] -> [N, C]
    targetsOneHot = identity.embedding(target);
  } else {
    targetsOneHot = target;
  }

  // -sum(target * log(probs)) / N
  return targetsOneHot.mul(logProbs).neg().sum(-1).mean();
}
import { Tensor } from "./engine/tensor";

/**
 * Creates a one-hot encoded tensor of given length, with 1 at index and 0 elsewhere.
 * @param index Index to set to 1
 * @param dims Length of the one-hot vector
 * @returns Tensor of shape [dims]
 */
export function oneHot(index: number, dims: number): Tensor {
  const tensor = Tensor.zeros([dims]);
  tensor.set([index], 1);
  return tensor;
}

/**
 * Create a batched one-hot tensor from an array of integer indices.
 * @param indices Array of length N with integer class indices
 * @param dims Number of classes (length of one-hot vector)
 * @returns Tensor of shape [N, dims]
 */
export function oneHotBatch(indices: number[], dims: number): Tensor {
  const N = indices.length;
  const data = new Float32Array(N * dims);
  for (let i = 0; i < N; i++) {
    const ix = indices[i];
    if (ix < 0 || ix >= dims) {
      throw new Error(`oneHotBatch: index ${ix} out of range [0, ${dims})`);
    }
    data[i * dims + ix] = 1;
  }
  return Tensor.fromData(data, [N, dims]);
}
