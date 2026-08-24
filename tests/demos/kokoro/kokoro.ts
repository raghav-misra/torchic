// Kokoro top-level: bert + bert_encoder + text_encoder + predictor +
// decoder from StyleTTS 2's build_model(), minus training-only pieces.
// forward() is a placeholder — wired at demo review with real weights.

import { Tensor } from "../../../src/frontend/tensor";
import { Module } from "../../../src/nn/module";
import { Linear } from "../../../src/nn/layers";
import { TextEncoder } from "./text_encoder";
import { ProsodyPredictor } from "./predictor";
import { Decoder } from "./istftnet";
import { PLBERT } from "./plbert";
import type { KokoroConfig } from "./config";
import { KOKORO_CONFIG } from "./config";

export interface TensorStats { shape: number[]; mean: number; std: number; min: number; max: number }

async function tensorStats(t: Tensor): Promise<TensorStats> {
  const flat = await t.toArray();
  let sum = 0, sq = 0, mn = Infinity, mx = -Infinity;
  for (const v of flat) {
    sum += v;
    sq += v * v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const n = flat.length;
  const mean = sum / n;
  return { shape: t.shape.slice(), mean, std: Math.sqrt(Math.max(0, sq / n - mean * mean)), min: mn, max: mx };
}

export class Kokoro extends Module {
  bert: PLBERT;
  bert_encoder: Linear;
  predictor: ProsodyPredictor;
  text_encoder: TextEncoder;
  decoder: Decoder;
  config: KokoroConfig;

  constructor(cfg: KokoroConfig = KOKORO_CONFIG) {
    super();
    this.config = cfg;
    this.bert = this.child(
      "bert",
      new PLBERT({
        vocabSize: cfg.n_token,
        hiddenSize: cfg.plbert.hidden_size,
        numHeads: cfg.plbert.num_attention_heads,
        intermediate: cfg.plbert.intermediate_size,
        maxPos: cfg.plbert.max_position_embeddings,
        numLayers: cfg.plbert.num_hidden_layers,
      }),
    );
    this.bert_encoder = this.child("bert_encoder", new Linear(cfg.plbert.hidden_size, cfg.hidden_dim));
    this.predictor = this.child(
      "predictor",
      new ProsodyPredictor(cfg.style_dim, cfg.hidden_dim, cfg.n_layer, cfg.max_dur),
    );
    this.text_encoder = this.child(
      "text_encoder",
      new TextEncoder(cfg.hidden_dim, cfg.text_encoder_kernel_size, cfg.n_layer, cfg.n_token),
    );
    this.decoder = this.child("decoder", new Decoder(cfg.istftnet, cfg.hidden_dim, cfg.style_dim));
  }

  // Full inference pass. input_ids: [1, T] phoneme ids (with wrapper zeros).
  // ref_s: [1, 256] voice pack row — first 128 = decoder style, second 128 = prosody style.
  // Returns raw PCM audio at 24 kHz plus the per-phoneme predicted durations.
  async forward(
    inputIds: Tensor,
    refS: Tensor,
    opts: {
      speed?: number;
      maxDurPerPhoneme?: number;
      onStats?: (name: string, stats: TensorStats) => void;
      onStage?: (name: string) => void;
    } = {},
  ): Promise<{ audio: Float32Array; predDur: number[]; rawDur: number[] }> {
    if (inputIds.shape.length !== 2) {
      throw new Error(`Kokoro.forward: input_ids must be [B, T], got ${inputIds.shape}`);
    }
    if (inputIds.shape[0] !== 1) {
      throw new Error(`Kokoro.forward: B=1 only, got ${inputIds.shape[0]}`);
    }
    const speed = opts.speed ?? 1;
    const maxDur = opts.maxDurPerPhoneme ?? Infinity;
    const stage = opts.onStage ?? ((_: string) => undefined);
    const dumpStats = async (name: string, t: Tensor): Promise<void> => {
      if (!opts.onStats) return;
      opts.onStats(name, await tensorStats(t));
    };

    const B = 1;
    const T = inputIds.shape[1];
    const styleDim = this.config.style_dim;

    const sDecoder = refS.slice([[0, B], [0, styleDim]]);
    const sProsody = refS.slice([[0, B], [styleDim, 2 * styleDim]]);

    stage("bert");
    const bertOut = await this.bert.forward(inputIds, undefined, async (i, h) => {
      const name = i === -1 ? "embeddings" : `bert_layer_${i}`;
      if (opts.onStats) opts.onStats(name, await tensorStats(h));
    });
    await dumpStats("bert_out", bertOut);
    stage("bert_encoder");
    const bertEnc = this.bert_encoder.forward(bertOut);
    const dEn = bertEnc.transpose(-1, -2);
    await dumpStats("d_en", dEn);
    bertOut.dispose();

    stage("dur_text_encoder");
    const d = this.predictor.text_encoder.forward(dEn, sProsody);
    await dumpStats("d (DurEnc out)", d);
    dEn.dispose();
    bertEnc.dispose();
    stage("dur_lstm");
    const dLstm = this.predictor.lstm.forward(d);
    await dumpStats("x (LSTM out)", dLstm);
    stage("dur_proj");
    const durationLogits = this.predictor.duration_proj.forward(dLstm);
    await dumpStats("duration_logits", durationLogits);
    dLstm.dispose();

    stage("dur_readback");
    const durationSum = durationLogits.sigmoid().sum(-1);
    const durationArr = await durationSum.toArray();
    durationLogits.dispose();
    durationSum.dispose();
    const predDur: number[] = [];
    const rawDur: number[] = [];
    for (let t = 0; t < T; t++) {
      const rawScaled = durationArr[t] / speed;
      rawDur.push(rawScaled);
      const rounded = Math.max(1, Math.round(rawScaled));
      predDur.push(Math.min(rounded, maxDur));
    }

    const L = predDur.reduce((a, b) => a + b, 0);
    stage(`aln_build (T=${T} L=${L})`);
    const alnData = new Float32Array(T * L);
    let l = 0;
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < predDur[t]; k++) {
        alnData[t * L + l] = 1;
        l++;
      }
    }
    const predAlnTrg = Tensor.fromData(Array.from(alnData), [B, T, L]);

    stage("aln_bmm_d");
    const dT = d.transpose(-1, -2);
    const en = dT.bmm(predAlnTrg);
    dT.dispose();
    d.dispose();
    stage("f0n_forward");
    const { F0, N } = this.predictor.F0Nforward(en, sProsody);
    en.dispose();

    stage("text_encoder");
    const tEn = this.text_encoder.forward(inputIds);
    stage("aln_bmm_tEn");
    const asr = tEn.bmm(predAlnTrg);
    tEn.dispose();
    predAlnTrg.dispose();

    stage("decoder");
    const audio = await this.decoder.forward(asr, F0, N, sDecoder, opts.onStage);
    asr.dispose();
    F0.dispose();
    N.dispose();
    sDecoder.dispose();
    sProsody.dispose();
    stage("done");
    return { audio, predDur, rawDur };
  }
}

// Total tensor element count across the state_dict. Sanity check for a
// loaded checkpoint or an in-flight skeleton.
export function countParameters(m: Module): number {
  const sd = m.state_dict();
  let total = 0;
  for (const key of Object.keys(sd)) {
    total += sd[key].shape.reduce((a, b) => a * b, 1);
  }
  return total;
}
