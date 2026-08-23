// PLBERT: Phoneme-Level ALBERT text encoder used by Kokoro.
// Ref: https://github.com/hexgrad/kokoro/blob/main/kokoro/modules.py (CustomAlbert)
// State_dict layout matches HuggingFace transformers.AlbertModel exactly.
//
// ALBERT vs BERT: embeddings live at embedding_size (128), get projected to
// hidden_size (768) once via encoder.embedding_hidden_mapping_in, then a
// SINGLE transformer layer is reused num_hidden_layers times. That's what
// gets the 84M-param BERT-base config down to 11M for ALBERT-base.

import { Tensor } from "../../../src/frontend/tensor";
import { Module } from "../../../src/nn/module";
import { Embedding, LayerNorm, Linear } from "../../../src/nn/layers";

export interface PLBERTOptions {
  vocabSize: number;
  hiddenSize: number;
  numHeads: number;
  intermediate: number;
  maxPos: number;
  numLayers: number;
  embeddingSize?: number;
  typeVocabSize?: number;
}

export class PLBERTEmbeddings extends Module {
  word_embeddings: Embedding;
  position_embeddings: Embedding;
  token_type_embeddings: Embedding;
  LayerNorm: LayerNorm;
  private embeddingSize: number;

  constructor(vocabSize: number, embeddingSize: number, maxPos: number, typeVocab: number) {
    super();
    this.embeddingSize = embeddingSize;
    this.word_embeddings = this.child("word_embeddings", new Embedding(vocabSize, embeddingSize));
    this.position_embeddings = this.child("position_embeddings", new Embedding(maxPos, embeddingSize));
    this.token_type_embeddings = this.child("token_type_embeddings", new Embedding(typeVocab, embeddingSize));
    this.LayerNorm = this.child("LayerNorm", new LayerNorm(embeddingSize));
  }

  // input_ids: [B, T] int -> [B, T, embedding_size]
  forward(inputIds: Tensor): Tensor {
    const [_B, T] = inputIds.shape;
    const positions = Tensor.fromData(Array.from({ length: T }, (_, i) => i), [T]);
    const tokenTypes = Tensor.fromData(new Array(T).fill(0), [T]);
    const word = this.word_embeddings.forward(inputIds);
    const pos = this.position_embeddings.forward(positions).reshape([1, T, this.embeddingSize]);
    const tt = this.token_type_embeddings.forward(tokenTypes).reshape([1, T, this.embeddingSize]);
    positions.dispose();
    tokenTypes.dispose();
    const wordPos = word.add(pos);
    word.dispose();
    pos.dispose();
    const summed = wordPos.add(tt);
    wordPos.dispose();
    tt.dispose();
    const out = this.LayerNorm.forward(summed);
    summed.dispose();
    return out;
  }
}

// One transformer block: pre-LN attention with an OUT projection + LN, then
// FFN + LN. Structure matches HF AlbertLayer.
export class AlbertLayer extends Module {
  attention: AlbertAttention;
  ffn: Linear;
  ffn_output: Linear;
  full_layer_layer_norm: LayerNorm;

  constructor(hiddenSize: number, numHeads: number, intermediate: number) {
    super();
    this.attention = this.child("attention", new AlbertAttention(hiddenSize, numHeads));
    this.ffn = this.child("ffn", new Linear(hiddenSize, intermediate));
    this.ffn_output = this.child("ffn_output", new Linear(intermediate, hiddenSize));
    this.full_layer_layer_norm = this.child("full_layer_layer_norm", new LayerNorm(hiddenSize));
  }

  forward(x: Tensor, attentionMask?: Tensor): Tensor {
    const attn = this.attention.forward(x, attentionMask);
    const ffnIn = this.ffn.forward(attn);
    const ffnAct = ffnIn.gelu();
    ffnIn.dispose();
    const ffn = this.ffn_output.forward(ffnAct);
    ffnAct.dispose();
    const summed = attn.add(ffn);
    attn.dispose();
    ffn.dispose();
    const out = this.full_layer_layer_norm.forward(summed);
    summed.dispose();
    return out;
  }
}

export class AlbertAttention extends Module {
  query: Linear;
  key: Linear;
  value: Linear;
  dense: Linear;
  LayerNorm: LayerNorm;

  private numHeads: number;
  private headDim: number;
  private hiddenSize: number;
  private scale: Tensor;

  constructor(hiddenSize: number, numHeads: number) {
    super();
    if (hiddenSize % numHeads !== 0) {
      throw new Error(`AlbertAttention: hiddenSize ${hiddenSize} not divisible by numHeads ${numHeads}`);
    }
    this.numHeads = numHeads;
    this.headDim = hiddenSize / numHeads;
    this.hiddenSize = hiddenSize;
    this.scale = Tensor.fromData([1 / Math.sqrt(this.headDim)]);
    this.query = this.child("query", new Linear(hiddenSize, hiddenSize));
    this.key = this.child("key", new Linear(hiddenSize, hiddenSize));
    this.value = this.child("value", new Linear(hiddenSize, hiddenSize));
    this.dense = this.child("dense", new Linear(hiddenSize, hiddenSize));
    this.LayerNorm = this.child("LayerNorm", new LayerNorm(hiddenSize));
  }

  forward(x: Tensor, _attentionMask?: Tensor): Tensor {
    const [B, T, H] = x.shape;
    const nH = this.numHeads;
    const dH = this.headDim;
    if (H !== this.hiddenSize) throw new Error(`AlbertAttention: got H=${H}, expected ${this.hiddenSize}`);

    const splitHeads = (t: Tensor): Tensor => {
      const r1 = t.reshape([B, T, nH, dH]);
      const trans = r1.transpose(1, 2);
      const r2 = trans.reshape([B * nH, T, dH]);
      t.dispose();
      return r2;
    };

    const q = splitHeads(this.query.forward(x));
    const k = splitHeads(this.key.forward(x));
    const v = splitHeads(this.value.forward(x));

    const kT = k.transpose(-1, -2);
    const qk = q.bmm(kT);
    q.dispose();
    k.dispose();
    const scores = qk.mul(this.scale);
    qk.dispose();

    const scoresFlat = scores.reshape([B * nH * T, T]);
    scores.dispose();
    const softmaxed = scoresFlat.softmax(-1);
    scoresFlat.dispose();
    const attn = softmaxed.reshape([B * nH, T, T]);
    softmaxed.dispose();

    const av = attn.bmm(v);
    attn.dispose();
    v.dispose();
    const avR = av.reshape([B, nH, T, dH]);
    av.dispose();
    const avT = avR.transpose(1, 2);
    const context = avT.reshape([B, T, H]);
    avR.dispose();

    const projected = this.dense.forward(context);
    context.dispose();
    const summed = x.add(projected);
    projected.dispose();
    const out = this.LayerNorm.forward(summed);
    summed.dispose();
    return out;
  }
}

// ALBERT groups layers so that N repeated layers share ONE set of weights.
// Kokoro uses num_hidden_groups=1 (implicit), so we always have exactly one
// group holding one layer, and the top-level encoder just calls that layer
// num_hidden_layers times.
export class AlbertLayerGroup extends Module {
  albert_layers: AlbertLayer[];

  constructor(hiddenSize: number, numHeads: number, intermediate: number) {
    super();
    this.albert_layers = this.childList("albert_layers", [
      new AlbertLayer(hiddenSize, numHeads, intermediate),
    ]);
  }

  forward(x: Tensor, mask?: Tensor): Tensor {
    return this.albert_layers[0].forward(x, mask);
  }
}

export class AlbertTransformer extends Module {
  embedding_hidden_mapping_in: Linear;
  albert_layer_groups: AlbertLayerGroup[];

  private numLayers: number;

  constructor(embeddingSize: number, hiddenSize: number, numLayers: number, numHeads: number, intermediate: number) {
    super();
    this.numLayers = numLayers;
    this.embedding_hidden_mapping_in = this.child(
      "embedding_hidden_mapping_in",
      new Linear(embeddingSize, hiddenSize),
    );
    this.albert_layer_groups = this.childList("albert_layer_groups", [
      new AlbertLayerGroup(hiddenSize, numHeads, intermediate),
    ]);
  }

  forward(x: Tensor, mask?: Tensor): Tensor {
    let h = this.embedding_hidden_mapping_in.forward(x);
    for (let i = 0; i < this.numLayers; i++) {
      const next = this.albert_layer_groups[0].forward(h, mask);
      h.dispose();
      h = next;
    }
    return h;
  }
}

export class PLBERT extends Module {
  embeddings: PLBERTEmbeddings;
  encoder: AlbertTransformer;
  // ALBERT ships a pooler in its state_dict even when the caller (Kokoro's
  // CustomAlbert) only uses last_hidden_state. Kept here for strict loading.
  pooler: Linear;

  constructor(opts: PLBERTOptions) {
    super();
    const embeddingSize = opts.embeddingSize ?? 128;
    const typeVocab = opts.typeVocabSize ?? 2;
    this.embeddings = this.child(
      "embeddings",
      new PLBERTEmbeddings(opts.vocabSize, embeddingSize, opts.maxPos, typeVocab),
    );
    this.encoder = this.child(
      "encoder",
      new AlbertTransformer(embeddingSize, opts.hiddenSize, opts.numLayers, opts.numHeads, opts.intermediate),
    );
    this.pooler = this.child("pooler", new Linear(opts.hiddenSize, opts.hiddenSize));
  }

  // input_ids: [B, T] -> last_hidden_state: [B, T, hidden_size]
  forward(inputIds: Tensor, attentionMask?: Tensor): Tensor {
    const embedded = this.embeddings.forward(inputIds);
    const out = this.encoder.forward(embedded, attentionMask);
    embedded.dispose();
    return out;
  }
}
