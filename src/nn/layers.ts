import { Tensor } from "../frontend/tensor";
import { Module } from "./module";
import { scaledRandn, kaimingStd } from "./init";

export class Linear extends Module {
  W: Tensor;
  b: Tensor | null;

  constructor(inDim: number, outDim: number, bias = true) {
    super();
    this.W = this.param("W", scaledRandn([inDim, outDim], kaimingStd(inDim)));
    this.b = bias ? this.param("b", Tensor.zeros([outDim], true)) : null;
  }

  forward(x: Tensor): Tensor {
    const out = x.matmul(this.W);
    return this.b ? out.add(this.b) : out;
  }
}

export class Embedding extends Module {
  W: Tensor;

  constructor(numEmbeddings: number, embedDim: number, initStd = 0.02) {
    super();
    this.W = this.param("W", scaledRandn([numEmbeddings, embedDim], initStd));
  }

  forward(indices: Tensor): Tensor {
    return this.W.embedding(indices);
  }
}

export class Sequential extends Module {
  layers: Module[];

  constructor(...layers: Module[]) {
    super();
    this.layers = this.childList("layers", layers);
  }

  forward(x: Tensor): Tensor {
    let h = x;
    for (const l of this.layers) {
      const fwd = (l as Module & { forward: (x: Tensor) => Tensor }).forward;
      h = fwd.call(l, h);
    }
    return h;
  }
}

// Layer normalization over the last `normalizedShape` dimensions. Composed from
// primitives — a fused kernel can replace this later without touching callers.
export class LayerNorm extends Module {
  weight: Tensor;
  bias: Tensor;
  private eps: Tensor;
  private normalizedShape: number[];

  constructor(normalizedShape: number | number[], eps = 1e-5) {
    super();
    const shape = typeof normalizedShape === "number" ? [normalizedShape] : normalizedShape;
    this.normalizedShape = shape;
    this.weight = this.param("weight", Tensor.ones(shape, true));
    this.bias = this.param("bias", Tensor.zeros(shape, true));
    this.eps = Tensor.fromData([eps]);
  }

  forward(x: Tensor): Tensor {
    if (this.normalizedShape.length !== 1) {
      throw new Error(
        `LayerNorm only supports 1-D normalizedShape for now, got ${this.normalizedShape}`,
      );
    }
    const axis = x.shape.length - 1;
    const mean = x.mean(axis, true);
    const centered = x.sub(mean);
    const variance = centered.mul(centered).mean(axis, true);
    const invStd = variance.add(this.eps).rsqrt();
    return centered.mul(invStd).mul(this.weight).add(this.bias);
  }
}
