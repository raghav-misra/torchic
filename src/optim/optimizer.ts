import { Tensor } from "../frontend/tensor";

export abstract class Optimizer {
  constructor(protected params: Tensor[]) {}

  abstract step(): Promise<void>;

  zeroGrad(): void {
    for (const p of this.params) p.grad = null;
  }
}
