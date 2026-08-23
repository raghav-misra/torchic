// Style-modulated normalization from StyleTTS 2. Both variants project a
// style vector `s: [B, style_dim]` through a Linear to (gamma, beta), then
// apply `(1 + gamma) * norm(x) + beta` channel-wise.
//
// Reference: https://github.com/yl4579/StyleTTS2/blob/main/models.py
//   class AdaIN1d(nn.Module):     # x: [B, C, L]
//   class AdaLayerNorm(nn.Module): # x: [B, C, T]

import { Tensor } from "../../frontend/tensor";
import { Module } from "../../nn/module";
import { Linear, InstanceNorm1d } from "../../nn/layers";

export class AdaIN1d extends Module {
  norm: InstanceNorm1d;
  fc: Linear;

  constructor(styleDim: number, numFeatures: number) {
    super();
    this.norm = this.child("norm", new InstanceNorm1d(numFeatures, 1e-5, false));
    this.fc = this.child("fc", new Linear(styleDim, numFeatures * 2));
  }

  // x: [B, C, L], style: [B, styleDim] -> [B, C, L]
  forward(x: Tensor, style: Tensor): Tensor {
    const [B, C, _L] = x.shape;
    const h = this.fc.forward(style).reshape([B, C * 2, 1]);
    const gamma = h.slice([[0, B], [0, C], [0, 1]]);
    const beta = h.slice([[0, B], [C, 2 * C], [0, 1]]);
    const one = Tensor.fromData([1]);
    return one.add(gamma).mul(this.norm.forward(x)).add(beta);
  }
}

// AdaLayerNorm normalizes over the channel dim (per-token layer norm), then
// applies the style modulation. In StyleTTS 2 the input to forward() is
// [B, T, C] (batch-first), and the output is [B, T, C].
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
    const [B, T, C] = x.shape;
    if (C !== this.channels) throw new Error(`AdaLayerNorm: C=${C} != channels=${this.channels}`);

    // Layer-norm over the last (channel) axis.
    const mean = x.mean(-1, true);
    const centered = x.sub(mean);
    const variance = centered.mul(centered).mean(-1, true);
    const invStd = variance.add(this.eps).rsqrt();
    const normed = centered.mul(invStd);

    const h = this.fc.forward(style).reshape([B, 1, 2 * C]);
    const gamma = h.slice([[0, B], [0, 1], [0, C]]);
    const beta = h.slice([[0, B], [0, 1], [C, 2 * C]]);
    const one = Tensor.fromData([1]);
    // Broadcast [B, 1, C] over T.
    return one.add(gamma).mul(normed).add(beta);
    void T;
  }
}
