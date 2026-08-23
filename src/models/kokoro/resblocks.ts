// AdainResBlk1d: Conv1d residual block modulated by AdaIN. Used by both
// ProsodyPredictor's F0/N stacks and the ISTFTNet decoder.
// Ref: https://github.com/yl4579/StyleTTS2/blob/main/models.py
//
// `upsample=true` inserts a grouped ConvTranspose1d(k=3, s=2) that doubles L.
// nn.ConvTranspose1d doesn't have `groups` yet — pin down at demo review.

import { Tensor } from "../../frontend/tensor";
import { Module } from "../../nn/module";
import { Conv1d, ConvTranspose1d } from "../../nn/layers";
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
    // Grouped ConvTranspose1d — one filter per channel — for the upsample path.
    // Groups aren't a native param on our nn.ConvTranspose1d yet; leave as null
    // when upsample is false. Kokoro's ProsodyPredictor uses upsample=true on
    // exactly one block per stack; the demo review is a good spot to add
    // grouped convs to the primitive if needed.
    this.pool = upsample
      ? this.child("pool", new ConvTranspose1d(dimIn, dimIn, 3, { stride: 2, padding: 1, outputPadding: 1 }))
      : null;
  }

  private shortcut(x: Tensor): Tensor {
    let out = x;
    if (this.pool) out = this.pool.forward(out);
    if (this.conv1x1) out = this.conv1x1.forward(out);
    return out;
  }

  private residual(x: Tensor, s: Tensor): Tensor {
    let out = this.norm1.forward(x, s).leaky_relu(this.actvSlope);
    if (this.pool) out = this.pool.forward(out);
    out = this.conv1.forward(out);
    out = this.norm2.forward(out, s).leaky_relu(this.actvSlope);
    out = this.conv2.forward(out);
    return out;
  }

  forward(x: Tensor, s: Tensor): Tensor {
    const r = this.residual(x, s);
    const sc = this.shortcut(x);
    return r.add(sc).mul(this.invSqrt2);
  }
}
