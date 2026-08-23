// TextEncoder from StyleTTS 2: phoneme embedding -> stack of (Conv1d,
// LayerNorm, LeakyReLU, Dropout) blocks -> BiLSTM. In Kokoro-82M:
// channels=hidden_dim=512, kernel_size=5, depth=n_layer=3, n_symbols=178.
//
// PyTorch reference:
//   class TextEncoder(nn.Module):
//     self.embedding = nn.Embedding(n_symbols, channels)
//     self.cnn = ModuleList of Sequential(Conv1d(k=5, p=2), LayerNorm(chn),
//                                         LeakyReLU(0.2), Dropout(0.2))
//     self.lstm = LSTM(chn, chn//2, 1, batch_first=True, bidirectional=True)
//
// The LayerNorm in StyleTTS 2 operates channels-first via a transpose sandwich.
// State_dict keys use the standard LSTM layout so BiLSTM's forward+bwd map to
// {weight_ih_l0, weight_hh_l0, ...} and {weight_ih_l0_reverse, ...}.
// (Our BiLSTM stores them under fwd./bwd. — see renameMap in kokoro.ts.)

import { Tensor } from "../../frontend/tensor";
import { Module } from "../../nn/module";
import { Conv1d, Embedding, BiLSTM, LayerNorm } from "../../nn/layers";

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
