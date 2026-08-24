// AdainResBlk1d: Conv1d residual block modulated by AdaIN. Used by both
// ProsodyPredictor's F0/N stacks and the ISTFTNet decoder.
// Ref: https://github.com/yl4579/StyleTTS2/blob/main/models.py
//
// `upsample=true` inserts a grouped ConvTranspose1d(k=3, s=2) that doubles L.
// nn.ConvTranspose1d doesn't have `groups` yet — pin down at demo review.

import { Tensor } from "../../../src/frontend/tensor";
import { Module } from "../../../src/nn/module";
import { Conv1d, ConvTranspose1d } from "../../../src/nn/layers";
import { AdaIN1d } from "./adain";

const SQRT2 = Math.sqrt(2);

export interface AdainResBlk1dOptions {
  upsample?: boolean;
  actvSlope?: number;
  dropoutP?: number; // ignored in inference mode
}

export class AdainResBlk1d extends Module {
  norm1: AdaIN1d;
  norm2: AdaIN1d;
  conv1: Conv1d;
  conv2: Conv1d;
  conv1x1: Conv1d | null;
  pool: ConvTranspose1d | null;
  private learnedSc: boolean;
  private actvSlope: number;
  private invSqrt2: Tensor;

  constructor(
    dimIn: number,
    dimOut: number,
    styleDim: number,
    opts: AdainResBlk1dOptions = {},
  ) {
    super();
    const upsample = opts.upsample ?? false;
    this.actvSlope = opts.actvSlope ?? 0.2;
    this.invSqrt2 = Tensor.fromData([1 / SQRT2]);
    this.learnedSc = dimIn !== dimOut;

    this.conv1 = this.child("conv1", new Conv1d(dimIn, dimOut, 3, { stride: 1, padding: 1 }));
    this.conv2 = this.child("conv2", new Conv1d(dimOut, dimOut, 3, { stride: 1, padding: 1 }));
    this.norm1 = this.child("norm1", new AdaIN1d(styleDim, dimIn));
    this.norm2 = this.child("norm2", new AdaIN1d(styleDim, dimOut));
    this.conv1x1 = this.learnedSc
      ? this.child("conv1x1", new Conv1d(dimIn, dimOut, 1, { stride: 1, padding: 0, bias: false }))
      : null;
    // Depthwise ConvTranspose1d — one filter per channel — for the 2x upsample path.
    this.pool = upsample
      ? this.child(
          "pool",
          new ConvTranspose1d(dimIn, dimIn, 3, {
            stride: 2,
            padding: 1,
            outputPadding: 1,
            groups: dimIn,
          }),
        )
      : null;
  }

  private shortcut(x: Tensor): Tensor {
    // Owns the returned tensor unless it's just `x` (caller keeps ownership).
    let out = x;
    if (this.pool) {
      const pooled = this.pool.forward(out);
      if (out !== x) out.dispose();
      out = pooled;
    }
    if (this.conv1x1) {
      const projected = this.conv1x1.forward(out);
      if (out !== x) out.dispose();
      out = projected;
    }
    return out;
  }

  private residual(x: Tensor, s: Tensor): Tensor {
    const normed1 = this.norm1.forward(x, s);
    let out = normed1.leaky_relu(this.actvSlope);
    normed1.dispose();
    if (this.pool) {
      const pooled = this.pool.forward(out);
      out.dispose();
      out = pooled;
    }
    const conv1Out = this.conv1.forward(out);
    out.dispose();
    const normed2 = this.norm2.forward(conv1Out, s);
    conv1Out.dispose();
    const activated2 = normed2.leaky_relu(this.actvSlope);
    normed2.dispose();
    const conv2Out = this.conv2.forward(activated2);
    activated2.dispose();
    return conv2Out;
  }

  forward(x: Tensor, s: Tensor): Tensor {
    const r = this.residual(x, s);
    const sc = this.shortcut(x);
    const added = r.add(sc);
    r.dispose();
    if (sc !== x) sc.dispose();
    const out = added.mul(this.invSqrt2);
    added.dispose();
    return out;
  }
}
