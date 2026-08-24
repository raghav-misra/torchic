import { Tensor, noGrad, noGradSync, trackTensors } from "./frontend/tensor";
import { oneHot, oneHotBatch, crossEntropy } from "./frontend/helpers";
import { init, shutdown, memoryStats, opCountSnapshot, resetOpCounts, sync } from "./frontend/dispatcher";

const torchic = {
  Tensor,
  oneHot,
  oneHotBatch,
  crossEntropy,
  init,
  shutdown,
  memoryStats,
  opCountSnapshot,
  resetOpCounts,
  sync,
};

export default torchic;

export { Tensor, noGrad, noGradSync, trackTensors, oneHot, oneHotBatch, crossEntropy, init, shutdown, memoryStats, opCountSnapshot, resetOpCounts, sync };
export * as nn from "./nn";
export * as optim from "./optim";
export * as dsp from "./dsp";
