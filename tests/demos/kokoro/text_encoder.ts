// TextEncoder from StyleTTS 2: phoneme embedding -> (Conv1d, LN,
// LeakyReLU) x depth -> BiLSTM. Kokoro-82M: 512 channels, k=5, depth=3.
// Ref: https://github.com/yl4579/StyleTTS2/blob/main/models.py

import { Tensor } from "../../../src/frontend/tensor";
import { Module } from "../../../src/nn/module";
import { Conv1d, Embedding, BiLSTM, LayerNorm } from "../../../src/nn/layers";

class ChannelsFirstLayerNorm extends Module {
  ln: LayerNorm;

  constructor(channels: number, eps = 1e-5) {
    super();
    this.ln = this.child("ln", new LayerNorm(channels, eps));
  }

  // x: [B, C, T] -> [B, C, T]
  forward(x: Tensor): Tensor {
    return this.ln.forward(x.transpose(1, 2)).transpose(1, 2);
  }
}

export class TextEncoder extends Module {
  embedding: Embedding;
  cnn: Module[];
  lstm: BiLSTM;
  private channels: number;

  constructor(channels: number, kernelSize: number, depth: number, nSymbols: number) {
    super();
    this.channels = channels;
    this.embedding = this.child("embedding", new Embedding(nSymbols, channels));
    const padding = (kernelSize - 1) >> 1;
    const blocks: Module[] = [];
    for (let i = 0; i < depth; i++) {
      // Each block is a Sequential-like triple, but the state_dict wants nested
      // keys `cnn.0.0.weight`, `cnn.0.1.weight`, ... — see PyTorch's Sequential.
      // We flatten via childList("cnn", blocks) below to get the right prefixes.
      blocks.push(new TextEncoderBlock(channels, kernelSize, padding));
    }
    this.cnn = this.childList("cnn", blocks);
    // BiLSTM output is [B, T, channels] (channels//2 per direction * 2).
    this.lstm = this.child("lstm", new BiLSTM(channels, channels / 2));
  }

  // x: [B, T] phoneme indices -> [B, T, channels]
  forward(x: Tensor): Tensor {
    // [B, T] -> [B, T, C] -> [B, C, T] for the conv stack.
    let h = this.embedding.forward(x).transpose(1, 2);
    for (const b of this.cnn) {
      h = (b as TextEncoderBlock).forward(h);
    }
    // Back to [B, T, C] for the BiLSTM.
    h = h.transpose(1, 2);
    return this.lstm.forward(h);
  }
}

class TextEncoderBlock extends Module {
  conv: Conv1d;
  norm: ChannelsFirstLayerNorm;

  constructor(channels: number, kernelSize: number, padding: number) {
    super();
    this.conv = this.child("conv", new Conv1d(channels, channels, kernelSize, { stride: 1, padding }));
    this.norm = this.child("norm", new ChannelsFirstLayerNorm(channels));
  }

  forward(x: Tensor): Tensor {
    return this.norm.forward(this.conv.forward(x)).leaky_relu(0.2);
  }
}
