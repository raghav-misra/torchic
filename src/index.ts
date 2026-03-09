import { Tensor, noGrad, trackTensors } from "./frontend/tensor";
import { oneHot, oneHotBatch, crossEntropy } from "./frontend/helpers";
import { init, shutdown } from "./frontend/dispatcher";

const torchic = {
  Tensor,
  oneHot,
  oneHotBatch,
  crossEntropy,
  init,
  shutdown,
};

export default torchic;

export { Tensor, noGrad, trackTensors, oneHot, oneHotBatch, crossEntropy, init, shutdown };
