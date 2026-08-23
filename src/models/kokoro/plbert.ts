// PLBERT: Phoneme-Level BERT text encoder used by Kokoro. Standard BERT
// layout (per-layer transformer weights, not ALBERT sharing).
// Kokoro-82M: 12 layers, hidden=768, heads=12, ff=2048, max_pos=512.
// Ref: https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py

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
