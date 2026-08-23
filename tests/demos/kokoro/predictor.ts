// ProsodyPredictor + DurationEncoder from StyleTTS 2. Kokoro-82M:
// style_dim=128, d_hid=512, n_layer=3, max_dur=50.
// Ref: https://github.com/yl4579/StyleTTS2/blob/main/models.py
//
// Reference implementation packs padded sequences for the LSTMs; at inference
// we always have a single utterance so we skip the pack/pad dance.

import { Tensor } from "../../../src/frontend/tensor";
import { Module } from "../../../src/nn/module";
import { BiLSTM, LinearNorm, Conv1d } from "../../../src/nn/layers";
import { AdaLayerNorm } from "./adain";
import { AdainResBlk1d } from "./resblocks";

export class DurationEncoder extends Module {
  lstms: Module[];
  private styDim: number;
  private dModel: number;

  constructor(styleDim: number, dModel: number, nLayers: number) {
    super();
    this.styDim = styleDim;
    this.dModel = dModel;
    const items: Module[] = [];
    for (let i = 0; i < nLayers; i++) {
      items.push(new BiLSTM(dModel + styleDim, dModel / 2));
      items.push(new AdaLayerNorm(styleDim, dModel));
    }
    this.lstms = this.childList("lstms", items);
  }

  // x: [B, dModel, T], style: [B, styleDim] -> [B, T, dModel + styleDim]
  // Style is concatenated once up front and re-concatenated after each
  // AdaLayerNorm block, so the final output carries style channels — matches
  // the reference's return format where predictor.lstm sees a 640-dim input.
  forward(x: Tensor, style: Tensor): Tensor {
    const [B, _C, T] = x.shape;
    const styleExpanded = this.expandStyle(style, B, T);
    let h = x.transpose(1, 2);
    h = Tensor.concat([h, styleExpanded], -1);
    for (const block of this.lstms) {
      if (block instanceof BiLSTM) {
        h = (block as BiLSTM).forward(h);
      } else {
        h = (block as AdaLayerNorm).forward(h, style);
        h = Tensor.concat([h, styleExpanded], -1);
      }
    }
    return h;
  }

  private expandStyle(style: Tensor, B: number, T: number): Tensor {
    // style: [B, styleDim] -> [B, T, styleDim] by tiling along the T axis.
    const parts: Tensor[] = [];
    const view = style.reshape([B, 1, this.styDim]);
    for (let t = 0; t < T; t++) parts.push(view);
    return Tensor.concat(parts, 1);
  }
}

export class ProsodyPredictor extends Module {
  text_encoder: DurationEncoder;
  lstm: BiLSTM;
  duration_proj: LinearNorm;
  shared: BiLSTM;
  F0: AdainResBlk1d[];
  N: AdainResBlk1d[];
  F0_proj: Conv1d;
  N_proj: Conv1d;

  constructor(styleDim: number, dHid: number, nLayers: number, maxDur: number) {
    super();
    this.text_encoder = this.child("text_encoder", new DurationEncoder(styleDim, dHid, nLayers));
    this.lstm = this.child("lstm", new BiLSTM(dHid + styleDim, dHid / 2));
    this.duration_proj = this.child("duration_proj", new LinearNorm(dHid, maxDur));
    this.shared = this.child("shared", new BiLSTM(dHid + styleDim, dHid / 2));
    const F0: AdainResBlk1d[] = [
      new AdainResBlk1d(dHid, dHid, styleDim),
      new AdainResBlk1d(dHid, dHid / 2, styleDim, { upsample: true }),
      new AdainResBlk1d(dHid / 2, dHid / 2, styleDim),
    ];
    const N: AdainResBlk1d[] = [
      new AdainResBlk1d(dHid, dHid, styleDim),
      new AdainResBlk1d(dHid, dHid / 2, styleDim, { upsample: true }),
      new AdainResBlk1d(dHid / 2, dHid / 2, styleDim),
    ];
    this.F0 = this.childList("F0", F0);
    this.N = this.childList("N", N);
    this.F0_proj = this.child("F0_proj", new Conv1d(dHid / 2, 1, 1, { stride: 1, padding: 0 }));
    this.N_proj = this.child("N_proj", new Conv1d(dHid / 2, 1, 1, { stride: 1, padding: 0 }));
  }

  // Predict per-phoneme duration + an "encoded" tensor for F0/N.
  // texts: [B, dModel, T] (already text-encoded)
  // style: [B, styleDim]
  // Returns:
  //   duration: [B, T, maxDur]
  //   d: [B, dModel + styleDim, T]  (for F0/N branch input after alignment expansion)
  forward(texts: Tensor, style: Tensor): { duration: Tensor; d: Tensor } {
    const d = this.text_encoder.forward(texts, style);
    const lstmOut = this.lstm.forward(d);
    const duration = this.duration_proj.forward(lstmOut);
    return { duration, d: d.transpose(1, 2) };
  }

  // F0Ntrain from the reference. x: [B, dHid + styleDim, L] (style baked in
  // by the caller's alignment expansion), s: [B, styleDim].
  // Returns F0, N each of shape [B, L*upsampleFactor].
  F0Nforward(x: Tensor, s: Tensor): { F0: Tensor; N: Tensor } {
    const sharedOut = this.shared.forward(x.transpose(1, 2));

    let F0 = sharedOut.transpose(1, 2);
    for (const block of this.F0) F0 = block.forward(F0, s);
    F0 = this.F0_proj.forward(F0);

    let N = sharedOut.transpose(1, 2);
    for (const block of this.N) N = block.forward(N, s);
    N = this.N_proj.forward(N);

    return {
      F0: F0.reshape([F0.shape[0], F0.shape[2]]),
      N: N.reshape([N.shape[0], N.shape[2]]),
    };
  }

  private expandStyle(style: Tensor, B: number, T: number): Tensor {
    const styleDim = style.shape[1];
    const parts: Tensor[] = [];
    const view = style.reshape([B, 1, styleDim]);
    for (let t = 0; t < T; t++) parts.push(view);
    return Tensor.concat(parts, 1);
  }
}
