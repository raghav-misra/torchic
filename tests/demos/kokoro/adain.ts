// Style-modulated normalization from StyleTTS 2: project style vector
// through a Linear to (gamma, beta), apply (1 + gamma) * norm(x) + beta.
// Ref: https://github.com/yl4579/StyleTTS2/blob/main/models.py

import { Tensor } from "../../../src/frontend/tensor";
import { Module } from "../../../src/nn/module";
import { Linear, InstanceNorm1d } from "../../../src/nn/layers";

export class AdaIN1d extends Module {
  norm: InstanceNorm1d;
  fc: Linear;

  constructor(styleDim: number, numFeatures: number) {
    super();
    // Reference kokoro/istftnet.py uses affine=True (see the comment there
    // about an ONNX-export bug), but the shipped checkpoint was trained with
    // affine=False so there are no `norm.weight`/`norm.bias` entries. Match
    // the checkpoint.
    this.norm = this.child("norm", new InstanceNorm1d(numFeatures, 1e-5, false));
    this.fc = this.child("fc", new Linear(styleDim, numFeatures * 2));
  }

  // x: [B, C, L], style: [B, styleDim] -> [B, C, L]
  forward(x: Tensor, style: Tensor): Tensor {
    const [B, C, _L] = x.shape;
    const fcOut = this.fc.forward(style);
    const h = fcOut.reshape([B, C * 2, 1]);
    const gamma = h.slice([[0, B], [0, C], [0, 1]]);
    const beta = h.slice([[0, B], [C, 2 * C], [0, 1]]);
    const one = Tensor.fromData([1]);
    const gammaPlus1 = one.add(gamma);
    one.dispose();
    const normX = this.norm.forward(x);
    const scaled = gammaPlus1.mul(normX);
    gammaPlus1.dispose();
    normX.dispose();
    const out = scaled.add(beta);
    scaled.dispose();
    gamma.dispose();
    beta.dispose();
    h.dispose();
    fcOut.dispose();
    return out;
  }
}

// LayerNorm-based counterpart to AdaIN1d. Operates on [B, T, C].
export class AdaLayerNorm extends Module {
  fc: Linear;
  private channels: number;
  private eps: Tensor;

  constructor(styleDim: number, channels: number, eps = 1e-5) {
    super();
    this.channels = channels;
    this.eps = Tensor.fromData([eps]);
    this.fc = this.child("fc", new Linear(styleDim, channels * 2));
  }

  // x: [B, T, C], style: [B, styleDim] -> [B, T, C]
  forward(x: Tensor, style: Tensor): Tensor {
    if (x.shape.length !== 3) throw new Error(`AdaLayerNorm expects [B, T, C], got ${x.shape}`);
    const [B, , C] = x.shape;
    if (C !== this.channels) throw new Error(`AdaLayerNorm: C=${C} != channels=${this.channels}`);

    const mean = x.mean(-1, true);
    const centered = x.sub(mean);
    mean.dispose();
    const sq = centered.mul(centered);
    const variance = sq.mean(-1, true);
    sq.dispose();
    const varPlusEps = variance.add(this.eps);
    variance.dispose();
    const invStd = varPlusEps.rsqrt();
    varPlusEps.dispose();
    const normed = centered.mul(invStd);
    centered.dispose();
    invStd.dispose();

    const fcOut = this.fc.forward(style);
    const h = fcOut.reshape([B, 1, 2 * C]);
    const gamma = h.slice([[0, B], [0, 1], [0, C]]);
    const beta = h.slice([[0, B], [0, 1], [C, 2 * C]]);
    const one = Tensor.fromData([1]);
    const gammaPlus1 = one.add(gamma);
    one.dispose();
    const scaled = gammaPlus1.mul(normed);
    gammaPlus1.dispose();
    normed.dispose();
    const out = scaled.add(beta);
    scaled.dispose();
    gamma.dispose();
    beta.dispose();
    h.dispose();
    fcOut.dispose();
    return out;
  }
}
