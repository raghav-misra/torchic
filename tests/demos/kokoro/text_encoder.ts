// TextEncoder from Kokoro (StyleTTS 2 lineage). Phoneme embedding -> depth ×
// (Conv1d, LayerNorm-gamma-beta, LeakyReLU, Dropout) -> BiLSTM.
// State_dict layout mirrors kokoro/modules.py: cnn.i.0 is the Conv1d,
// cnn.i.1 is a channels-first LayerNorm with gamma/beta params.
// Kokoro-82M: 512 channels, k=5, depth=3.

import { Tensor } from "../../../src/frontend/tensor";
import { Module } from "../../../src/nn/module";
import { Conv1d, Embedding, BiLSTM } from "../../../src/nn/layers";

// Kokoro's per-channel LayerNorm. Normalizes over the channel dim of [B, C, T],
// so it needs a transpose in and out. Uses gamma/beta names (not weight/bias),
// matching kokoro/modules.py so the checkpoint loads cleanly.
class KokoroChannelLN extends Module {
  gamma: Tensor;
  beta: Tensor;
  private channels: number;
  private eps: Tensor;

  constructor(channels: number, eps = 1e-5) {
    super();
    this.channels = channels;
    this.eps = Tensor.fromData([eps]);
    this.gamma = this.param("gamma", Tensor.ones([channels], true));
    this.beta = this.param("beta", Tensor.zeros([channels], true));
  }

  forward(x: Tensor): Tensor {
    // [B, C, T] -> [B, T, C] -> layer_norm on C -> transpose back.
    const t = x.transpose(1, 2);
    const mean = t.mean(-1, true);
    const centered = t.sub(mean);
    mean.dispose();
    t.dispose();
    const sq = centered.mul(centered);
    const variance = sq.mean(-1, true);
    sq.dispose();
    const varPlusEps = variance.add(this.eps);
    variance.dispose();
    const invStd = varPlusEps.rsqrt();
    varPlusEps.dispose();
    const scaled = centered.mul(invStd);
    centered.dispose();
    invStd.dispose();
    const gained = scaled.mul(this.gamma);
    scaled.dispose();
    const normed = gained.add(this.beta);
    gained.dispose();
    const out = normed.transpose(1, 2);
    // normed is the returned view's source; hand ownership over via out.prev.
    // Consumer disposes `out`, source stays alive via view ref.
    return out;
  }
}

// One CNN block from the reference's `nn.Sequential(Conv1d, LayerNorm,
// LeakyReLU, Dropout)`. LeakyReLU/Dropout have no params, so state_dict only
// exposes children at positions 0 and 1.
class TextEncoderCNNBlock extends Module {
  private conv: Conv1d;
  private norm: KokoroChannelLN;

  constructor(channels: number, kernelSize: number, padding: number) {
    super();
    this.conv = this.child("0", new Conv1d(channels, channels, kernelSize, { stride: 1, padding }));
    this.norm = this.child("1", new KokoroChannelLN(channels));
  }

  forward(x: Tensor): Tensor {
    const c = this.conv.forward(x);
    const n = this.norm.forward(c);
    c.dispose();
    const out = n.leaky_relu(0.2);
    n.dispose();
    return out;
  }
}

export class TextEncoder extends Module {
  embedding: Embedding;
  cnn: TextEncoderCNNBlock[];
  lstm: BiLSTM;
  private channels: number;

  constructor(channels: number, kernelSize: number, depth: number, nSymbols: number) {
    super();
    this.channels = channels;
    this.embedding = this.child("embedding", new Embedding(nSymbols, channels));
    const padding = (kernelSize - 1) >> 1;
    const blocks: TextEncoderCNNBlock[] = [];
    for (let i = 0; i < depth; i++) {
      blocks.push(new TextEncoderCNNBlock(channels, kernelSize, padding));
    }
    this.cnn = this.childList("cnn", blocks);
    this.lstm = this.child("lstm", new BiLSTM(channels, channels / 2));
  }

  // x: [B, T] phoneme indices -> [B, channels, T]
  forward(x: Tensor): Tensor {
    const emb = this.embedding.forward(x);
    let h = emb.transpose(1, 2);
    for (const b of this.cnn) {
      const next = b.forward(h);
      h.dispose();
      h = next;
    }
    emb.dispose();
    const hT = h.transpose(1, 2);
    h.dispose();
    const lstmOut = this.lstm.forward(hT);
    hT.dispose();
    const out = lstmOut.transpose(1, 2);
    return out;
  }
}
