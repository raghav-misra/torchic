// Kokoro top-level model. Wires the pieces from StyleTTS 2's build_model()
// minus training-only components (diffusion, discriminators, aligner).
//
// forward() will need voice pack + phoneme indices + alignment matrix. For now
// this class just constructs the module tree with the right shapes and enough
// forward wiring to point at where the demo review needs to verify parity.

import { Tensor } from "../../frontend/tensor";
import { Module } from "../../nn/module";
import { Linear } from "../../nn/layers";
import { TextEncoder } from "./text_encoder";
import { ProsodyPredictor } from "./predictor";
import { ISTFTGenerator } from "./istftnet";
import { PLBERT } from "./plbert";
import type { KokoroConfig } from "./config";
import { KOKORO_CONFIG } from "./config";

export class Kokoro extends Module {
  bert: PLBERT;
  bert_encoder: Linear;
  predictor: ProsodyPredictor;
  text_encoder: TextEncoder;
  decoder: ISTFTGenerator;
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
    // bert_encoder projects PLBERT hidden -> hidden_dim (512) for the predictor.
    this.bert_encoder = this.child("bert_encoder", new Linear(cfg.plbert.hidden_size, cfg.hidden_dim));
    this.predictor = this.child(
      "predictor",
      new ProsodyPredictor(cfg.style_dim, cfg.hidden_dim, cfg.n_layer, cfg.max_dur),
    );
    this.text_encoder = this.child(
      "text_encoder",
      new TextEncoder(cfg.hidden_dim, cfg.text_encoder_kernel_size, cfg.n_layer, cfg.n_token),
    );
    // decoder input channels = hidden_dim (from text_encoder), but the reference
    // decoder concatenates additional info (F0, N, style) before the first conv.
    // We match the reference's decoder input signature at demo-review time.
    this.decoder = this.child("decoder", new ISTFTGenerator(cfg.istftnet, cfg.hidden_dim));
  }

  // Placeholder end-to-end wiring. Real inference (see demo review) will:
  //  1. G2P: text -> phoneme indices
  //  2. bert(phonemes) -> features [B, T, 768]
  //  3. bert_encoder -> [B, T, 512]
  //  4. text_encoder(phonemes) -> [B, T, 512]  (parallel path)
  //  5. predictor.forward -> duration + d
  //  6. duration -> alignment matrix
  //  7. predictor.F0Nforward(d @ alignment, style) -> F0, N
  //  8. decoder(en, F0, N, style) -> waveform
  //
  // The pieces are all here; the demo review is where we bolt them together
  // against the real checkpoint.
  forward(_phonemeIds: Tensor, _style: Tensor): never {
    throw new Error(
      "Kokoro.forward is a demo-review placeholder. Wire this up together with the actual weights and voice pack.",
    );
  }
}

// Convenience: total param count. Handy at demo review to compare against
// the 82M target and catch missing modules.
export function countParameters(m: Module): number {
  const sd = m.state_dict();
  let total = 0;
  for (const key of Object.keys(sd)) {
    total += sd[key].shape.reduce((a, b) => a * b, 1);
  }
  return total;
}
