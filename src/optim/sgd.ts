import { Tensor, noGrad } from "../frontend/tensor";
import { Optimizer } from "./optimizer";

export class SGD extends Optimizer {
  private lrTensor: Tensor;

  constructor(params: Tensor[], lr: number) {
    super(params);
    this.lrTensor = Tensor.fromData([lr]);
  }

  async step(): Promise<void> {
    await noGrad(async () => {
      for (const p of this.params) {
        if (p.grad) p.sub_(p.grad.mul(this.lrTensor));
      }
    });
  }

  setLR(lr: number): void {
    this.lrTensor.set([0], lr);
  }
}
