// ISTFTNet decoder for Kokoro. HiFi-GAN Multi-Receptive-Field generator
// that emits magnitude + phase spectrograms, then ISTFTs to audio.
// Ref: https://github.com/yl4579/StyleTTS2/blob/main/Modules/istftnet.py

import { Tensor } from "../../../src/frontend/tensor";
import { Module } from "../../../src/nn/module";
import { Conv1d, ConvTranspose1d } from "../../../src/nn/layers";
import type { KokoroISTFTNetConfig } from "./config";
import { istft, hannWindow } from "../../../src/dsp";

const LEAKY_SLOPE = 0.1;

// HiFi-GAN ResBlock1: for each dilation d, two Conv1d(k, d)/Conv1d(k, 1) pairs
// with LeakyReLU in between, wrapped with a residual add.
export class ResBlock1 extends Module {
  convs1: Conv1d[];
  convs2: Conv1d[];

  constructor(channels: number, kernelSize: number, dilations: number[]) {
    super();
    const pad = (d: number) => ((kernelSize - 1) * d) >> 1;
    const c1 = dilations.map((d) =>
      new Conv1d(channels, channels, kernelSize, { stride: 1, padding: pad(d), dilation: d }),
    );
    const c2 = dilations.map(
      () => new Conv1d(channels, channels, kernelSize, { stride: 1, padding: pad(1), dilation: 1 }),
    );
    this.convs1 = this.childList("convs1", c1);
    this.convs2 = this.childList("convs2", c2);
  }

  forward(x: Tensor): Tensor {
    let h = x;
    for (let i = 0; i < this.convs1.length; i++) {
      const xt = h.leaky_relu(LEAKY_SLOPE);
      const c1o = this.convs1[i].forward(xt);
      const xt2 = c1o.leaky_relu(LEAKY_SLOPE);
      const c2o = this.convs2[i].forward(xt2);
      h = c2o.add(h);
    }
    return h;
  }
}

// Multi-Receptive-Field module: sum of several ResBlock1 outputs.
export class MRF extends Module {
  resblocks: ResBlock1[];

  constructor(channels: number, kernelSizes: number[], dilationSizes: number[][]) {
    super();
    const rbs: ResBlock1[] = [];
    for (let i = 0; i < kernelSizes.length; i++) {
      rbs.push(new ResBlock1(channels, kernelSizes[i], dilationSizes[i]));
    }
    this.resblocks = this.childList("resblocks", rbs);
  }

  forward(x: Tensor): Tensor {
    if (this.resblocks.length === 0) throw new Error("MRF: no resblocks");
    let acc = this.resblocks[0].forward(x);
    for (let i = 1; i < this.resblocks.length; i++) {
      acc = acc.add(this.resblocks[i].forward(x));
    }
    const inv = Tensor.fromData([1 / this.resblocks.length]);
    return acc.mul(inv);
  }
}

export class ISTFTGenerator extends Module {
  conv_pre: Conv1d;
  ups: ConvTranspose1d[];
  mrfs: MRF[];
  conv_post_mag: Conv1d;
  conv_post_phase: Conv1d;
  private nFFT: number;
  private hop: number;

  constructor(cfg: KokoroISTFTNetConfig, inChannels: number) {
    super();
    this.nFFT = cfg.gen_istft_n_fft;
    this.hop = cfg.gen_istft_hop_size;

    this.conv_pre = this.child(
      "conv_pre",
      new Conv1d(inChannels, cfg.upsample_initial_channel, 7, { stride: 1, padding: 3 }),
    );

    let C = cfg.upsample_initial_channel;
    const ups: ConvTranspose1d[] = [];
    const mrfs: MRF[] = [];
    for (let i = 0; i < cfg.upsample_rates.length; i++) {
      const r = cfg.upsample_rates[i];
      const k = cfg.upsample_kernel_sizes[i];
      const outC = C >> 1;
      ups.push(
        new ConvTranspose1d(C, outC, k, { stride: r, padding: (k - r) >> 1 }),
      );
      C = outC;
      mrfs.push(new MRF(C, cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes));
    }
    this.ups = this.childList("ups", ups);
    this.mrfs = this.childList("mrfs", mrfs);

    // Two projection heads: log-magnitude and phase (both have nFFT/2+1 bins).
    const nBins = (this.nFFT >> 1) + 1;
    this.conv_post_mag = this.child(
      "conv_post_mag",
      new Conv1d(C, nBins, 7, { stride: 1, padding: 3 }),
    );
    this.conv_post_phase = this.child(
      "conv_post_phase",
      new Conv1d(C, nBins, 7, { stride: 1, padding: 3 }),
    );
  }

  // x: [B, inChannels, T] -> (magnitude, phase) each [B, nBins, T_out]
  forwardSpec(x: Tensor): { magnitude: Tensor; phase: Tensor } {
    let h = this.conv_pre.forward(x);
    for (let i = 0; i < this.ups.length; i++) {
      h = h.leaky_relu(LEAKY_SLOPE);
      h = this.ups[i].forward(h);
      h = this.mrfs[i].forward(h);
    }
    h = h.leaky_relu(LEAKY_SLOPE);
    const magnitude = this.conv_post_mag.forward(h).exp(); // exp for log-magnitude -> magnitude
    const phase = this.conv_post_phase.forward(h);
    return { magnitude, phase };
  }

  // Full generator: [B, C, T] -> Float32Array audio (B=1 assumed for now).
  async forward(x: Tensor): Promise<Float32Array> {
    const { magnitude, phase } = this.forwardSpec(x);
    if (magnitude.shape[0] !== 1) {
      throw new Error(`ISTFTGenerator.forward supports B=1 for now, got B=${magnitude.shape[0]}`);
    }
    const magFlat = await magnitude.toArray();
    const phFlat = await phase.toArray();
    const nBins = magnitude.shape[1];
    const T = magnitude.shape[2];

    // Convert (magnitude, phase) -> complex, then ISTFT.
    const real = new Float32Array(nBins * T);
    const imag = new Float32Array(nBins * T);
    for (let i = 0; i < real.length; i++) {
      real[i] = magFlat[i] * Math.cos(phFlat[i]);
      imag[i] = magFlat[i] * Math.sin(phFlat[i]);
    }
    return istft(real, imag, T, {
      nFFT: this.nFFT,
      hopLength: this.hop,
      winLength: this.nFFT,
      window: hannWindow(this.nFFT),
      center: true,
    });
  }
}
