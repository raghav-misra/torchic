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
