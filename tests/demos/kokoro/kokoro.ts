// Kokoro top-level: bert + bert_encoder + text_encoder + predictor +
// decoder from StyleTTS 2's build_model(), minus training-only pieces.
// forward() is a placeholder — wired at demo review with real weights.

import { Tensor } from "../../../src/frontend/tensor";
import { Module } from "../../../src/nn/module";
import { Linear } from "../../../src/nn/layers";
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
    // Reference decoder concats (features, F0, N, style) before conv_pre; we
    // wire the concat at demo review once we see the real input signature.
    this.decoder = this.child("decoder", new ISTFTGenerator(cfg.istftnet, cfg.hidden_dim));
  }

  // Demo-review placeholder. The full pipeline is:
  //  1. text -> phoneme ids (G2P, main thread)
  //  2. bert -> bert_encoder -> [B, T, 512] features
  //  3. text_encoder -> parallel [B, T, 512] path
  //  4. predictor.forward -> per-phoneme duration logits
  //  5. durations -> alignment matrix (main thread)
  //  6. predictor.F0Nforward(features @ alignment, style) -> F0, N
  //  7. decoder(features, F0, N, style) -> audio
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
