// ProsodyPredictor + DurationEncoder from StyleTTS 2. Kokoro-82M:
// style_dim=128, d_hid=512, n_layer=3, max_dur=50.
// Ref: https://github.com/yl4579/StyleTTS2/blob/main/models.py
//
// Reference implementation packs padded sequences for the LSTMs; at inference
// we always have a single utterance so we skip the pack/pad dance.

import { Tensor } from "../../../src/frontend/tensor";
import { Module } from "../../../src/nn/module";
import { BiLSTM, Linear, Conv1d } from "../../../src/nn/layers";
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

  // x: [B, dModel, T], style: [B, styleDim] -> [B, T, dModel]
  forward(x: Tensor, style: Tensor): Tensor {
    // Broadcast style over T and concat channelwise before each BiLSTM.
    // Reference implementation packs/pads for the LSTM; we run at length T
    // directly (single-utterance inference).
    const [B, _C, T] = x.shape;
    let h = x.transpose(1, 2); // [B, T, C]
    const styleExpanded = this.expandStyle(style, B, T); // [B, T, styleDim]
    for (const block of this.lstms) {
      if (block instanceof BiLSTM) {
        const cat = Tensor.concat([h, styleExpanded], -1);
        h = (block as BiLSTM).forward(cat);
      } else {
        // AdaLayerNorm expects [B, T, C].
        h = (block as AdaLayerNorm).forward(h, style);
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
  duration_proj: Linear;
  shared: BiLSTM;
  F0: AdainResBlk1d[];
  N: AdainResBlk1d[];
  F0_proj: Conv1d;
  N_proj: Conv1d;

  constructor(styleDim: number, dHid: number, nLayers: number, maxDur: number) {
    super();
    this.text_encoder = this.child("text_encoder", new DurationEncoder(styleDim, dHid, nLayers));
    this.lstm = this.child("lstm", new BiLSTM(dHid + styleDim, dHid / 2));
    this.duration_proj = this.child("duration_proj", new Linear(dHid, maxDur));
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
  // texts: [B, dModel, T] (already text-encoded and style-catted)
  // style: [B, styleDim]
  // Returns:
  //   duration: [B, T, maxDur]  (log-duration logits; caller argmax + sum)
  //   d: [B, dModel, T]         (for F0/N branch input)
  forward(texts: Tensor, style: Tensor): { duration: Tensor; d: Tensor } {
    const d = this.text_encoder.forward(texts, style); // [B, T, dModel]
    // Concat style so LSTM input matches training layout.
    const [B, T, _dModel] = d.shape;
    const styleExpanded = this.expandStyle(style, B, T);
    const cat = Tensor.concat([d, styleExpanded], -1);
    const lstmOut = this.lstm.forward(cat); // [B, T, dModel]
    const duration = this.duration_proj.forward(lstmOut);
    // Return d in [B, dModel, T] for the F0/N stacks.
    return { duration, d: d.transpose(1, 2) };
  }

  // F0Ntrain from the reference — runs the shared BiLSTM + F0/N stacks.
  // x: [B, dModel, T'], s: [B, styleDim] -> ([B, T''], [B, T''])
  F0Nforward(x: Tensor, s: Tensor): { F0: Tensor; N: Tensor } {
    // shared LSTM operates on [B, T, dModel], so transpose in.
    const inp = x.transpose(1, 2); // [B, T, dModel]
    const styleExpanded = this.expandStyle(s, inp.shape[0], inp.shape[1]);
    const sharedOut = this.shared.forward(Tensor.concat([inp, styleExpanded], -1)); // [B, T, dModel]

    let F0 = sharedOut.transpose(1, 2);
    for (const block of this.F0) F0 = block.forward(F0, s);
    F0 = this.F0_proj.forward(F0);

    let N = sharedOut.transpose(1, 2);
    for (const block of this.N) N = block.forward(N, s);
    N = this.N_proj.forward(N);

    // Squeeze the singleton channel out.
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
