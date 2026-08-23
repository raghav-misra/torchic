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

  forward(_phonemeIds: Tensor, _style: Tensor): never {
    throw new Error("Kokoro.forward not wired yet — pending demo review with real weights.");
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
