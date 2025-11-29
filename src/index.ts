import { Tensor, noGrad, trackTensors } from "./frontend/tensor";
import { oneHot, crossEntropy } from "./frontend/helpers";
import { init, shutdown } from "./frontend/dispatcher";

const torchic = {
  Tensor,
  oneHot,
  crossEntropy,
  init,
  shutdown,
};

export default torchic;

export { Tensor, noGrad, trackTensors, oneHot, crossEntropy, init, shutdown };
