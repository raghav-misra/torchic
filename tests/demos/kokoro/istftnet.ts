// Kokoro decoder: ISTFTNet with F0-driven source-filter Generator.
// Ref: https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py
// State_dict layout matches the reference for direct HF checkpoint loading.

import { Tensor } from "../../../src/frontend/tensor";
import { Module } from "../../../src/nn/module";
import { Conv1d, ConvTranspose1d, Linear, Sequential } from "../../../src/nn/layers";
import type { KokoroISTFTNetConfig } from "./config";
import { istft, hannWindow } from "../../../src/dsp";
import { AdaIN1d } from "./adain";
import { AdainResBlk1d } from "./resblocks";

const LEAKY_SLOPE = 0.1;
const SAMPLE_RATE = 24000;

// Standard-normal sample via Box-Muller. Reference SineGen / SourceModuleHnNSF
// use torch.randn_like for their noise terms; a uniform in [-1, 1] has the
// wrong spectrum and audibly changes the noise floor.
function randn(): number {
  let u = 0;
  while (u === 0) u = Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function getPadding(kernelSize: number, dilation: number): number {
  return Math.floor(((kernelSize - 1) * dilation) / 2);
}

// Snake-based resblock used inside the Generator. Distinct from AdainResBlk1d,
// which is LeakyReLU-based and used by ProsodyPredictor + Decoder.encode/decode.
// State_dict layout: convs1[3], convs2[3], adain1[3], adain2[3], alpha1[3],
// alpha2[3] — three parallel branches with per-branch dilation on convs1.
export class AdaINResBlock1 extends Module {
  convs1: Conv1d[];
  convs2: Conv1d[];
  adain1: AdaIN1d[];
  adain2: AdaIN1d[];
  private alpha1: Tensor[];
  private alpha2: Tensor[];
  private channels: number;

  constructor(channels: number, kernelSize: number, dilations: number[], styleDim: number) {
    super();
    this.channels = channels;
    const c1 = dilations.map(
      (d) =>
        new Conv1d(channels, channels, kernelSize, {
          stride: 1,
          padding: getPadding(kernelSize, d),
          dilation: d,
        }),
    );
    const c2 = dilations.map(
      () =>
        new Conv1d(channels, channels, kernelSize, {
          stride: 1,
          padding: getPadding(kernelSize, 1),
          dilation: 1,
        }),
    );
    this.convs1 = this.childList("convs1", c1);
    this.convs2 = this.childList("convs2", c2);
    this.adain1 = this.childList(
      "adain1",
      dilations.map(() => new AdaIN1d(styleDim, channels)),
    );
    this.adain2 = this.childList(
      "adain2",
      dilations.map(() => new AdaIN1d(styleDim, channels)),
    );
    this.alpha1 = this.paramList(
      "alpha1",
      dilations.map(() => Tensor.ones([1, channels, 1], true)),
    );
    this.alpha2 = this.paramList(
      "alpha2",
      dilations.map(() => Tensor.ones([1, channels, 1], true)),
    );
  }

  private snake(x: Tensor, alpha: Tensor): Tensor {
    const ax = alpha.mul(x);
    const sinAx = ax.sin();
    ax.dispose();
    const sinSq = sinAx.mul(sinAx);
    sinAx.dispose();
    const ones = Tensor.ones([1, this.channels, 1]);
    const invAlpha = ones.div(alpha);
    ones.dispose();
    const scaled = invAlpha.mul(sinSq);
    invAlpha.dispose();
    sinSq.dispose();
    const out = x.add(scaled);
    scaled.dispose();
    return out;
  }

  forward(x: Tensor, s: Tensor): Tensor {
    let out = x;
    for (let i = 0; i < this.convs1.length; i++) {
      let xt = this.adain1[i].forward(out, s);
      const snaked1 = this.snake(xt, this.alpha1[i]);
      xt.dispose();
      xt = this.convs1[i].forward(snaked1);
      snaked1.dispose();
      const norm2 = this.adain2[i].forward(xt, s);
      xt.dispose();
      const snaked2 = this.snake(norm2, this.alpha2[i]);
      norm2.dispose();
      xt = this.convs2[i].forward(snaked2);
      snaked2.dispose();
      const next = xt.add(out);
      xt.dispose();
      if (out !== x) out.dispose();
      out = next;
    }
    return out;
  }
}

// F0-driven harmonic sine generator. Zero learnable params; the forward pass
// runs on the CPU as Float32Array manipulation because cumsum + rand are
// awkward on the current kernel set and the buffer is small (24kHz × seconds).
export class SineGen {
  private sampleRate: number;
  private harmonicNum: number;
  private sineAmp: number;
  private noiseStd: number;
  private voicedThreshold: number;

  constructor(
    sampleRate: number,
    _upsampleScale: number,
    harmonicNum = 0,
    sineAmp = 0.1,
    noiseStd = 0.003,
    voicedThreshold = 0,
  ) {
    this.sampleRate = sampleRate;
    this.harmonicNum = harmonicNum;
    this.sineAmp = sineAmp;
    this.noiseStd = noiseStd;
    this.voicedThreshold = voicedThreshold;
  }

  get dim(): number {
    return this.harmonicNum + 1;
  }

  // f0Values: [B, L] flat Float32Array. Returns sines [B, L, dim], uv [B, L].
  forward(f0Values: Float32Array, B: number, L: number): { sines: Float32Array; uv: Float32Array } {
    const dim = this.dim;
    const sines = new Float32Array(B * L * dim);
    const uv = new Float32Array(B * L);
    const twoPi = 2 * Math.PI;

    for (let b = 0; b < B; b++) {
      for (let h = 0; h < dim; h++) {
        const mult = h + 1;
        let phaseSum = 0;
        for (let l = 0; l < L; l++) {
          const rad = ((f0Values[b * L + l] * mult) / this.sampleRate) % 1;
          phaseSum += rad;
          sines[b * L * dim + l * dim + h] = Math.sin(phaseSum * twoPi) * this.sineAmp;
        }
      }
      for (let l = 0; l < L; l++) {
        uv[b * L + l] = f0Values[b * L + l] > this.voicedThreshold ? 1 : 0;
      }
    }
    for (let b = 0; b < B; b++) {
      for (let l = 0; l < L; l++) {
        const uvl = uv[b * L + l];
        const noiseAmp = uvl * this.noiseStd + (1 - uvl) * (this.sineAmp / 3);
        for (let h = 0; h < dim; h++) {
          const i = b * L * dim + l * dim + h;
          sines[i] = sines[i] * uvl + noiseAmp * randn();
        }
      }
    }
    return { sines, uv };
  }
}

// Merges harmonic sines through a small Linear + tanh; the caller sees a
// [B, L, 1] excitation and a [B, L, 1] noise stream.
export class SourceModuleHnNSF extends Module {
  l_linear: Linear;
  private l_sin_gen: SineGen;
  private sineAmp: number;

  constructor(
    sampleRate: number,
    upsampleScale: number,
    harmonicNum = 0,
    sineAmp = 0.1,
    noiseStd = 0.003,
    voicedThreshold = 0,
  ) {
    super();
    this.sineAmp = sineAmp;
    this.l_sin_gen = new SineGen(sampleRate, upsampleScale, harmonicNum, sineAmp, noiseStd, voicedThreshold);
    this.l_linear = this.child("l_linear", new Linear(harmonicNum + 1, 1));
  }

  // x: [B, L, 1] upsampled F0 curve.
  async forward(x: Tensor): Promise<{ sineMerge: Tensor; noise: Tensor; uv: Tensor }> {
    const [B, L] = x.shape;
    const f0 = await x.toArray();
    const { sines, uv } = this.l_sin_gen.forward(f0, B, L);
    const dim = this.l_sin_gen.dim;

    const sinesT = Tensor.fromData(Array.from(sines), [B, L, dim]);
    const sineMerge = this.l_linear.forward(sinesT).tanh();

    const uvT = Tensor.fromData(Array.from(uv), [B, L, 1]);
    const noiseData = new Float32Array(B * L);
    for (let i = 0; i < noiseData.length; i++) {
      noiseData[i] = randn() * (this.sineAmp / 3);
    }
    const noise = Tensor.fromData(Array.from(noiseData), [B, L, 1]);
    return { sineMerge, noise, uv: uvT };
  }
}

// Main-thread STFT of a real signal using a Hann window. Returns magnitude
// and phase per frame. Matches the reference TorchSTFT (return_complex=True
// path collapsed to (abs, angle)).
function stftHann(
  x: Float32Array,
  nFFT: number,
  hop: number,
): { magnitude: Float32Array; phase: Float32Array; frames: number } {
  const win = hannWindow(nFFT);
  // Center=True padding with reflection so the first frame is centered on t=0.
  const padded = new Float32Array(x.length + nFFT);
  const half = nFFT >> 1;
  for (let i = 0; i < half; i++) padded[i] = x[Math.min(half - i, x.length - 1)];
  for (let i = 0; i < x.length; i++) padded[half + i] = x[i];
  for (let i = 0; i < half; i++) {
    const src = x.length - 2 - i;
    padded[half + x.length + i] = src >= 0 ? x[src] : 0;
  }

  const nBins = (nFFT >> 1) + 1;
  const frames = Math.floor((padded.length - nFFT) / hop) + 1;
  const magnitude = new Float32Array(nBins * frames);
  const phase = new Float32Array(nBins * frames);
  const buf = new Float32Array(nFFT * 2);

  for (let f = 0; f < frames; f++) {
    const off = f * hop;
    for (let n = 0; n < nFFT; n++) {
      buf[2 * n] = padded[off + n] * win[n];
      buf[2 * n + 1] = 0;
    }
    // Direct DFT (nFFT is 20 for Kokoro; O(nFFT²) is cheap enough).
    for (let k = 0; k < nBins; k++) {
      let re = 0;
      let im = 0;
      for (let n = 0; n < nFFT; n++) {
        const angle = (-2 * Math.PI * k * n) / nFFT;
        re += buf[2 * n] * Math.cos(angle);
        im += buf[2 * n] * Math.sin(angle);
      }
      magnitude[k * frames + f] = Math.sqrt(re * re + im * im);
      phase[k * frames + f] = Math.atan2(im, re);
    }
  }
  return { magnitude, phase, frames };
}

// Kokoro's Generator: F0-driven source-filter vocoder with ISTFT tail.
// Layout follows kokoro/istftnet.py Generator:
//   m_source, ups[N], noise_convs[N], noise_res[N], resblocks[N*num_kernels], conv_post.
export class Generator extends Module {
  m_source: SourceModuleHnNSF;
  ups: ConvTranspose1d[];
  noise_convs: Conv1d[];
  noise_res: AdaINResBlock1[];
  resblocks: AdaINResBlock1[];
  conv_post: Conv1d;

  private numKernels: number;
  private numUpsamples: number;
  private postNFFT: number;
  private hopLength: number;
  private upsampleScale: number;

  constructor(cfg: KokoroISTFTNetConfig, styleDim: number) {
    super();
    this.numKernels = cfg.resblock_kernel_sizes.length;
    this.numUpsamples = cfg.upsample_rates.length;
    this.postNFFT = cfg.gen_istft_n_fft;
    this.hopLength = cfg.gen_istft_hop_size;
    this.upsampleScale = cfg.upsample_rates.reduce((a, b) => a * b, 1) * cfg.gen_istft_hop_size;

    this.m_source = this.child(
      "m_source",
      new SourceModuleHnNSF(SAMPLE_RATE, this.upsampleScale, 8, 0.1, 0.003, 10),
    );

    const ups: ConvTranspose1d[] = [];
    for (let i = 0; i < cfg.upsample_rates.length; i++) {
      const u = cfg.upsample_rates[i];
      const k = cfg.upsample_kernel_sizes[i];
      const inC = cfg.upsample_initial_channel >> i;
      const outC = cfg.upsample_initial_channel >> (i + 1);
      ups.push(new ConvTranspose1d(inC, outC, k, { stride: u, padding: (k - u) >> 1 }));
    }
    this.ups = this.childList("ups", ups);

    const noiseConvs: Conv1d[] = [];
    const noiseRes: AdaINResBlock1[] = [];
    const resblocks: AdaINResBlock1[] = [];
    for (let i = 0; i < cfg.upsample_rates.length; i++) {
      const ch = cfg.upsample_initial_channel >> (i + 1);
      for (let j = 0; j < cfg.resblock_kernel_sizes.length; j++) {
        resblocks.push(
          new AdaINResBlock1(ch, cfg.resblock_kernel_sizes[j], cfg.resblock_dilation_sizes[j], styleDim),
        );
      }
      if (i + 1 < cfg.upsample_rates.length) {
        const strideF0 = cfg.upsample_rates.slice(i + 1).reduce((a, b) => a * b, 1);
        noiseConvs.push(
          new Conv1d(cfg.gen_istft_n_fft + 2, ch, strideF0 * 2, {
            stride: strideF0,
            padding: (strideF0 + 1) >> 1,
          }),
        );
        noiseRes.push(new AdaINResBlock1(ch, 7, [1, 3, 5], styleDim));
      } else {
        noiseConvs.push(new Conv1d(cfg.gen_istft_n_fft + 2, ch, 1));
        noiseRes.push(new AdaINResBlock1(ch, 11, [1, 3, 5], styleDim));
      }
    }
    this.noise_convs = this.childList("noise_convs", noiseConvs);
    this.noise_res = this.childList("noise_res", noiseRes);
    this.resblocks = this.childList("resblocks", resblocks);

    const finalCh = cfg.upsample_initial_channel >> cfg.upsample_rates.length;
    this.conv_post = this.child(
      "conv_post",
      new Conv1d(finalCh, cfg.gen_istft_n_fft + 2, 7, { stride: 1, padding: 3 }),
    );
  }

  // x: [B, C, T], s: [B, styleDim], f0: [B, T_f0]. Returns Float32Array audio (B=1).
  async forward(x: Tensor, s: Tensor, f0: Tensor): Promise<Float32Array> {
    const [B] = f0.shape;
    if (B !== 1) throw new Error(`Generator: B=1 only for now, got ${B}`);

    // Upsample F0 by prod(upsample_rates)*hop, generate harmonic source.
    const f0Flat = await f0.toArray();
    const T_f0 = f0.shape[1];
    const L_source = T_f0 * this.upsampleScale;
    const upsampled = new Float32Array(L_source);
    for (let i = 0; i < L_source; i++) upsampled[i] = f0Flat[Math.floor(i / this.upsampleScale)];
    const upT = Tensor.fromData(Array.from(upsampled), [B, L_source, 1]);
    const { sineMerge } = await this.m_source.forward(upT);
    // sineMerge: [B, L_source, 1] -> [B, L_source]
    const harSource = await sineMerge.toArray();
    const { magnitude, phase, frames } = stftHann(harSource, this.postNFFT, this.hopLength);
    const nBins = (this.postNFFT >> 1) + 1;
    // Interleave [mag; phase] along channel axis -> [1, 2*nBins, frames]
    const harData = new Float32Array(2 * nBins * frames);
    harData.set(magnitude, 0);
    harData.set(phase, nBins * frames);
    const har = Tensor.fromData(Array.from(harData), [B, 2 * nBins, frames]);

    let h = x;
    for (let i = 0; i < this.numUpsamples; i++) {
      const hRelu = h.leaky_relu(LEAKY_SLOPE);
      if (h !== x) h.dispose();

      let xSource = this.noise_convs[i].forward(har);
      const xSourceRes = this.noise_res[i].forward(xSource, s);
      xSource.dispose();
      xSource = xSourceRes;

      let hUp = this.ups[i].forward(hRelu);
      hRelu.dispose();
      if (i === this.numUpsamples - 1) {
        const padded = hUp.reflectionPad1d(1, 0);
        hUp.dispose();
        hUp = padded;
      }
      const hAdd = hUp.add(xSource);
      hUp.dispose();
      xSource.dispose();

      let xs: Tensor | null = null;
      for (let j = 0; j < this.numKernels; j++) {
        const block = this.resblocks[i * this.numKernels + j].forward(hAdd, s);
        if (xs === null) {
          xs = block;
        } else {
          const summed: Tensor = xs.add(block);
          xs.dispose();
          block.dispose();
          xs = summed;
        }
      }
      hAdd.dispose();
      const inv = Tensor.fromData([1 / this.numKernels]);
      h = xs!.mul(inv);
      xs!.dispose();
      inv.dispose();
    }
    // Reference uses F.leaky_relu(x) — DEFAULT slope 0.01 — for this final
    // activation, not the 0.1 used inside the upsample loop.
    const hFinal = h.leaky_relu(0.01);
    h.dispose();
    h = this.conv_post.forward(hFinal);
    hFinal.dispose();
    har.dispose();

    // Split channel axis: first nBins = log-mag, second nBins = phase.
    const specPart = h.slice([[0, B], [0, nBins], [0, h.shape[2]]]).exp();
    const phasePart = h.slice([[0, B], [nBins, 2 * nBins], [0, h.shape[2]]]).sin();

    const specFlat = await specPart.toArray();
    const phaseFlat = await phasePart.toArray();
    const T = h.shape[2];
    const real = new Float32Array(nBins * T);
    const imag = new Float32Array(nBins * T);
    for (let i = 0; i < real.length; i++) {
      real[i] = specFlat[i] * Math.cos(phaseFlat[i]);
      imag[i] = specFlat[i] * Math.sin(phaseFlat[i]);
    }
    return istft(real, imag, T, {
      nFFT: this.postNFFT,
      hopLength: this.hopLength,
      winLength: this.postNFFT,
      window: hannWindow(this.postNFFT),
      center: true,
    });
  }
}

// Kokoro's Decoder: concat (asr, F0, N) -> AdainResBlk1d encode -> 4 decode
// blocks (each re-concats residual + F0 + N) -> Generator.
export class Decoder extends Module {
  encode: AdainResBlk1d;
  decode: AdainResBlk1d[];
  F0_conv: Conv1d;
  N_conv: Conv1d;
  asr_res: Sequential;
  generator: Generator;

  constructor(cfg: KokoroISTFTNetConfig, hiddenDim: number, styleDim: number) {
    super();
    this.encode = this.child("encode", new AdainResBlk1d(hiddenDim + 2, 1024, styleDim));
    this.decode = this.childList("decode", [
      new AdainResBlk1d(1024 + 2 + 64, 1024, styleDim),
      new AdainResBlk1d(1024 + 2 + 64, 1024, styleDim),
      new AdainResBlk1d(1024 + 2 + 64, 1024, styleDim),
      new AdainResBlk1d(1024 + 2 + 64, 512, styleDim, { upsample: true }),
    ]);
    this.F0_conv = this.child("F0_conv", new Conv1d(1, 1, 3, { stride: 2, padding: 1 }));
    this.N_conv = this.child("N_conv", new Conv1d(1, 1, 3, { stride: 2, padding: 1 }));
    this.asr_res = this.child(
      "asr_res",
      new Sequential(new Conv1d(hiddenDim, 64, 1, { stride: 1, padding: 0 })),
    );
    this.generator = this.child("generator", new Generator(cfg, styleDim));
  }

  // asr: [B, hiddenDim, L], F0_curve: [B, 2L], N: [B, 2L], s: [B, styleDim]
  async forward(asr: Tensor, F0_curve: Tensor, N: Tensor, s: Tensor): Promise<Float32Array> {
    const F0In = F0_curve.reshape([F0_curve.shape[0], 1, F0_curve.shape[1]]);
    const NIn = N.reshape([N.shape[0], 1, N.shape[1]]);
    const F0 = this.F0_conv.forward(F0In);
    const N2 = this.N_conv.forward(NIn);

    let x = Tensor.concat([asr, F0, N2], 1);
    x = this.encode.forward(x, s);
    const asrRes = this.asr_res.forward(asr);

    let addRes = true;
    for (const block of this.decode) {
      if (addRes) x = Tensor.concat([x, asrRes, F0, N2], 1);
      x = block.forward(x, s);
      if (block["pool"] !== null) addRes = false;
    }
    return this.generator.forward(x, s, F0_curve);
  }
}
