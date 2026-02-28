export function embedding(
  weights: Float32Array,
  indices: Float32Array,
  output: Float32Array,
  embeddingDim: number,
  start: number,
  end: number,
) {
  for (let i = start; i < end; i++) {
    const indexInIndices = Math.floor(i / embeddingDim);
    const offsetInEmbedding = i % embeddingDim;
    const row = indices[indexInIndices];
    output[i] = weights[row * embeddingDim + offsetInEmbedding];
  }
}

export function embedding_backward(
  weightsGrad: Float32Array,
  indices: Float32Array,
  outputGrad: Float32Array,
  embeddingDim: number,
  start: number,
  end: number,
) {
  // Non-atomic scatter-add — must be dispatched to a single worker
  for (let i = start; i < end; i++) {
    const indexInIndices = Math.floor(i / embeddingDim);
    const offsetInEmbedding = i % embeddingDim;
    const row = indices[indexInIndices];
    weightsGrad[row * embeddingDim + offsetInEmbedding] += outputGrad[i];
  }
}
