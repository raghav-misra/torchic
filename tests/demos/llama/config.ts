export interface LlamaConfig {
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  ffnSize: number;
  vocabSize: number;
  ropeTheta: number;
  rmsEps: number;
  tiedEmbeddings: boolean;
}

export const LLAMA_3_2_1B: LlamaConfig = {
  hiddenSize: 2048,
  numLayers: 16,
  numHeads: 32,
  numKvHeads: 8,
  headDim: 64,
  ffnSize: 8192,
  vocabSize: 128256,
  ropeTheta: 500_000,
  rmsEps: 1e-5,
  tiedEmbeddings: true,
};
