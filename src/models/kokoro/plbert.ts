// PLBERT: Phoneme-Level BERT (ALBERT-style parameter sharing) used as the
// text encoder feature extractor in Kokoro. Reference:
//   https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
//
// Kokoro-82M config:
//   hidden_size=768, num_attention_heads=12, intermediate_size=2048,
//   max_position_embeddings=512, num_hidden_layers=12, dropout=0.1
//
// ALBERT shares the transformer layer's weights across all N layers (they
// call it one "layer group"). We follow that: one TransformerEncoderLayer,
// applied num_hidden_layers times.
//
// The state_dict prefix in Kokoro is `bert.` for this whole module and
// `bert_encoder.` for the Linear that projects into hidden_dim=512.

import { Tensor } from "../../frontend/tensor";
import { Module } from "../../nn/module";
import {
  Embedding,
  LayerNorm,
  TransformerEncoderLayer,
} from "../../nn/layers";

export interface PLBERTOptions {
  vocabSize: number;
  hiddenSize: number;
  numHeads: number;
  intermediate: number;
  maxPos: number;
  numLayers: number;
  typeVocabSize?: number; // BERT-style; ALBERT typically has 2 (defaults to 1)
}

export class PLBERTEmbeddings extends Module {
  word_embeddings: Embedding;
  position_embeddings: Embedding;
  token_type_embeddings: Embedding;
  LayerNorm: LayerNorm;

  constructor(vocabSize: number, hiddenSize: number, maxPos: number, typeVocab: number) {
    super();
    this.word_embeddings = this.child("word_embeddings", new Embedding(vocabSize, hiddenSize));
    this.position_embeddings = this.child(
      "position_embeddings",
      new Embedding(maxPos, hiddenSize),
    );
    this.token_type_embeddings = this.child(
      "token_type_embeddings",
      new Embedding(typeVocab, hiddenSize),
    );
    this.LayerNorm = this.child("LayerNorm", new LayerNorm(hiddenSize));
  }

  // input_ids: [B, T] int -> [B, T, hidden]
  forward(inputIds: Tensor): Tensor {
    const [_B, T] = inputIds.shape;
    const positions = Tensor.fromData(
      Array.from({ length: T }, (_, i) => i),
      [T],
    );
    const tokenTypes = Tensor.fromData(new Array(T).fill(0), [T]);
    const word = this.word_embeddings.forward(inputIds);
    const pos = this.position_embeddings.forward(positions);
    const tt = this.token_type_embeddings.forward(tokenTypes);
    // Broadcast: pos [T, H] + tt [T, H] -> [1, T, H] before add-to-batch.
    const embedded = word.add(pos.reshape([1, T, pos.shape[1]])).add(tt.reshape([1, T, tt.shape[1]]));
    return this.LayerNorm.forward(embedded);
  }
}

export class PLBERT extends Module {
  embeddings: PLBERTEmbeddings;
  layers: TransformerEncoderLayer[];

  constructor(opts: PLBERTOptions) {
    super();
    const typeVocab = opts.typeVocabSize ?? 2;
    this.embeddings = this.child(
      "embeddings",
      new PLBERTEmbeddings(opts.vocabSize, opts.hiddenSize, opts.maxPos, typeVocab),
    );
    const layers: TransformerEncoderLayer[] = [];
    for (let i = 0; i < opts.numLayers; i++) {
      layers.push(
        new TransformerEncoderLayer(opts.hiddenSize, opts.numHeads, opts.intermediate / opts.hiddenSize),
      );
    }
    this.layers = this.childList("encoder.layer", layers);
  }

  forward(inputIds: Tensor): Tensor {
    let h = this.embeddings.forward(inputIds);
    for (const layer of this.layers) h = layer.forward(h);
    return h;
  }
}
