import { Tensor } from "../../../src/index";
import { Module } from "../../../src/nn/module";
import { Linear, Embedding, RMSNorm } from "../../../src/nn/layers";
import { KVCache } from "../../../src/nn/kv_cache";
import * as functional from "../../../src/nn/functional";
import type { LlamaConfig } from "./config";

// GQA self-attention with RoPE and KV cache streaming. Single-batch (B=1);
// input/output shape [T, hiddenSize]. Weight-name shape matches HF Llama:
// q_proj/k_proj/v_proj/o_proj as bias-less Linear children.
export class LlamaAttention extends Module {
  q_proj: Linear;
  k_proj: Linear;
  v_proj: Linear;
  o_proj: Linear;
  private readonly config: LlamaConfig;
  private readonly gqaGroups: number;

  constructor(config: LlamaConfig) {
    super();
    this.config = config;
    this.gqaGroups = config.numHeads / config.numKvHeads;
    const H = config.hiddenSize;
    const qOut = config.numHeads * config.headDim;
    const kvOut = config.numKvHeads * config.headDim;
    this.q_proj = this.child("q_proj", new Linear(H, qOut, false));
    this.k_proj = this.child("k_proj", new Linear(H, kvOut, false));
    this.v_proj = this.child("v_proj", new Linear(H, kvOut, false));
    this.o_proj = this.child("o_proj", new Linear(qOut, H, false));
  }

  forward(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    cache: KVCache,
    layerIdx: number,
    pastLen: number,
  ): Tensor {
    const T = x.shape[0];
    const nh = this.config.numHeads;
    const nkv = this.config.numKvHeads;
    const d = this.config.headDim;

    // Project and split into per-head format: [T, H] → [nh, T, d] (Q) or [nkv, T, d] (K/V).
    const q = this.q_proj.forward(x).reshape([T, nh, d]).transpose(0, 1);
    const k = this.k_proj.forward(x).reshape([T, nkv, d]).transpose(0, 1);
    const v = this.v_proj.forward(x).reshape([T, nkv, d]).transpose(0, 1);

    // RoPE at positions [pastLen, pastLen+T) on Q and K.
    const cosSlice = cos.slice([
      [pastLen, pastLen + T],
      [0, d / 2],
    ]);
    const sinSlice = sin.slice([
      [pastLen, pastLen + T],
      [0, d / 2],
    ]);
    const qRoped = q.rope(cosSlice, sinSlice);
    const kRoped = k.rope(cosSlice, sinSlice);

    // Cache stores [T, nkv, d]; transpose back before writing, and get views of
    // the [pastLen+T, nkv, d] context (which we then transpose to [nkv, Tk, d]
    // for attention).
    const kForCache = kRoped.transpose(0, 1);
    const vForCache = v.transpose(0, 1);
    const { k: kFull, v: vFull } = cache.write(layerIdx, kForCache, vForCache, T);
    const kAttn = kFull.transpose(0, 1);
    const vAttn = vFull.transpose(0, 1);

    // GQA broadcast: each of the 8 KV heads is shared by gqaGroups Q heads.
    const kBcast = kAttn.repeatInterleave(0, this.gqaGroups);
    const vBcast = vAttn.repeatInterleave(0, this.gqaGroups);

    const ctx = functional.causalAttention(qRoped, kBcast, vBcast, pastLen);

    // Merge heads: [nh, T, d] → [T, nh, d] → [T, nh*d].
    const merged = ctx.transpose(0, 1).reshape([T, nh * d]);
    return this.o_proj.forward(merged);
  }
}

// Wraps SwiGLU with HF-Llama child names (gate_proj/up_proj/down_proj).
export class LlamaMLP extends Module {
  gate_proj: Linear;
  up_proj: Linear;
  down_proj: Linear;

  constructor(config: LlamaConfig) {
    super();
    this.gate_proj = this.child("gate_proj", new Linear(config.hiddenSize, config.ffnSize, false));
    this.up_proj = this.child("up_proj", new Linear(config.hiddenSize, config.ffnSize, false));
    this.down_proj = this.child("down_proj", new Linear(config.ffnSize, config.hiddenSize, false));
  }

  forward(x: Tensor): Tensor {
    const gated = this.gate_proj.forward(x).silu();
    const up = this.up_proj.forward(x);
    return this.down_proj.forward(gated.mul(up));
  }
}

export class LlamaDecoderLayer extends Module {
  input_layernorm: RMSNorm;
  self_attn: LlamaAttention;
  post_attention_layernorm: RMSNorm;
  mlp: LlamaMLP;

  constructor(config: LlamaConfig) {
    super();
    this.input_layernorm = this.child("input_layernorm", new RMSNorm(config.hiddenSize, config.rmsEps));
    this.self_attn = this.child("self_attn", new LlamaAttention(config));
    this.post_attention_layernorm = this.child(
      "post_attention_layernorm",
      new RMSNorm(config.hiddenSize, config.rmsEps),
    );
    this.mlp = this.child("mlp", new LlamaMLP(config));
  }

  forward(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    cache: KVCache,
    layerIdx: number,
    pastLen: number,
  ): Tensor {
    const attnIn = this.input_layernorm.forward(x);
    const attnOut = this.self_attn.forward(attnIn, cos, sin, cache, layerIdx, pastLen);
    const afterAttn = x.add(attnOut);

    const mlpIn = this.post_attention_layernorm.forward(afterAttn);
    const mlpOut = this.mlp.forward(mlpIn);
    return afterAttn.add(mlpOut);
  }
}

// Named "model" to match HF Llama's outer wrapper (weights are under model.*).
export class LlamaModel extends Module {
  embed_tokens: Embedding;
  layers: LlamaDecoderLayer[];
  norm: RMSNorm;
  private readonly ropeCos: Tensor;
  private readonly ropeSin: Tensor;
  readonly config: LlamaConfig;

  constructor(config: LlamaConfig, maxSeqLen: number) {
    super();
    this.config = config;
    this.embed_tokens = this.child(
      "embed_tokens",
      new Embedding(config.vocabSize, config.hiddenSize),
    );
    // childList emits state_dict keys as `layers.<i>.<subpath>` matching HF.
    this.layers = this.childList(
      "layers",
      Array.from({ length: config.numLayers }, () => new LlamaDecoderLayer(config)),
    );
    this.norm = this.child("norm", new RMSNorm(config.hiddenSize, config.rmsEps));

    const { cos, sin } = functional.precomputeRope(maxSeqLen, config.headDim, config.ropeTheta);
    this.ropeCos = cos;
    this.ropeSin = sin;
  }

  // Runs the full stack and returns per-layer hidden states plus the
  // post-final-norm output, in the same order HF's `output_hidden_states=True`
  // uses (index 0 = embed output; index i>=1 = layer i-1 output; last = post-norm).
  forward(tokenIds: Tensor, cache: KVCache, pastLen: number): Tensor[] {
    const dumps: Tensor[] = [];
    let x = this.embed_tokens.forward(tokenIds);
    dumps.push(x);
    for (let i = 0; i < this.layers.length; i++) {
      x = this.layers[i].forward(x, this.ropeCos, this.ropeSin, cache, i, pastLen);
      dumps.push(x);
    }
    const postNorm = this.norm.forward(x);
    dumps.push(postNorm);
    return dumps;
  }
}

// Tied LM head: reuses embed_tokens.weight for the final projection. HF Llama
// stores no separate lm_head tensor when tie_word_embeddings=True.
export class LlamaForCausalLM extends Module {
  model: LlamaModel;

  constructor(config: LlamaConfig, maxSeqLen: number) {
    super();
    this.model = this.child("model", new LlamaModel(config, maxSeqLen));
  }

  forward(tokenIds: Tensor, cache: KVCache, pastLen: number): { hiddens: Tensor[]; logits: Tensor } {
    const hiddens = this.model.forward(tokenIds, cache, pastLen);
    const postNorm = hiddens[hiddens.length - 1];
    const embedW = this.model.embed_tokens.W;
    const T = postNorm.shape[0];
    const logits = postNorm.matmul(embedW.transpose(-1, -2));
    return { hiddens, logits };
  }
}
