import { Tensor } from "../frontend/tensor";

// Normal-distributed init with a chosen stddev. Used for Linear (Kaiming),
// Embedding (small normal), etc. Fresh tensor, safe to scale in-place.
export function scaledRandn(shape: number[], std: number, requiresGrad = true): Tensor {
  const t = Tensor.randn(shape, requiresGrad);
  if (std !== 1) {
    const s = Tensor.fromData([std]);
    t.mul_(s);
  }
  return t;
}

// Kaiming (He) stddev for fan-in `n`: sqrt(2/n). Matches PyTorch default for Linear.
export function kaimingStd(fanIn: number): number {
  return Math.sqrt(2 / fanIn);
}
