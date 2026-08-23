import { Tensor } from "../frontend/tensor";
import { Module } from "./module";
import { scaledRandn, kaimingStd } from "./init";

export class Linear extends Module {
  W: Tensor;
  b: Tensor | null;
  private inDim: number;
  private outDim: number;

  constructor(inDim: number, outDim: number, bias = true) {
    super();
    this.inDim = inDim;
    this.outDim = outDim;
    this.W = this.param("W", scaledRandn([inDim, outDim], kaimingStd(inDim)));
    this.b = bias ? this.param("b", Tensor.zeros([outDim], true)) : null;
  }

  forward(x: Tensor): Tensor {
    const rank = x.shape.length;
    if (x.shape[rank - 1] !== this.inDim) {
      throw new Error(
        `Linear: last dim ${x.shape[rank - 1]} != inDim ${this.inDim} (shape ${x.shape})`,
      );
    }
    // PyTorch convention: matmul over the last dim, broadcast over leading dims.
    if (rank === 2) {
      const out = x.matmul(this.W);
      return this.b ? out.add(this.b) : out;
    }
    const leading = x.shape.slice(0, rank - 1);
    const flat = x.reshape([-1, this.inDim]);
    const out2d = flat.matmul(this.W);
    const out = out2d.reshape([...leading, this.outDim]);
    return this.b ? out.add(this.b) : out;
  }
}

export class Embedding extends Module {
  W: Tensor;

  constructor(numEmbeddings: number, embedDim: number, initStd = 0.02) {
    super();
    this.W = this.param("W", scaledRandn([numEmbeddings, embedDim], initStd));
  }

  forward(indices: Tensor): Tensor {
    return this.W.embedding(indices);
  }
}

export class Sequential extends Module {
  layers: Module[];

  constructor(...layers: Module[]) {
    super();
    this.layers = this.childList("layers", layers);
  }

  forward(x: Tensor): Tensor {
    let h = x;
    for (const l of this.layers) {
      const fwd = (l as Module & { forward: (x: Tensor) => Tensor }).forward;
      h = fwd.call(l, h);
    }
    return h;
  }
}

// Layer normalization over the last `normalizedShape` dimensions. Composed from
// primitives — a fused kernel can replace this later without touching callers.
export class LayerNorm extends Module {
  weight: Tensor;
  bias: Tensor;
  private eps: Tensor;
  private normalizedShape: number[];

  constructor(normalizedShape: number | number[], eps = 1e-5) {
    super();
    const shape = typeof normalizedShape === "number" ? [normalizedShape] : normalizedShape;
    this.normalizedShape = shape;
    this.weight = this.param("weight", Tensor.ones(shape, true));
    this.bias = this.param("bias", Tensor.zeros(shape, true));
    this.eps = Tensor.fromData([eps]);
  }

  forward(x: Tensor): Tensor {
    if (this.normalizedShape.length !== 1) {
      throw new Error(
        `LayerNorm only supports 1-D normalizedShape for now, got ${this.normalizedShape}`,
      );
    }
    const axis = x.shape.length - 1;
    const mean = x.mean(axis, true);
    const centered = x.sub(mean);
    const variance = centered.mul(centered).mean(axis, true);
    const invStd = variance.add(this.eps).rsqrt();
    return centered.mul(invStd).mul(this.weight).add(this.bias);
  }
}

// Standard scaled dot-product multi-head attention over [B, S, D].
// Composes from Linear + reshape/transpose + bmm + softmax. Attention pattern:
//   Q, K, V = W_{q,k,v}(x)           # each [B, S, D]
//   split into H heads              # [B, H, S, D_h]
//   scores = Q @ K^T / sqrt(D_h)     # [B, H, S, S]
//   attn = softmax(scores, dim=-1)
//   out = W_o(concat_heads(attn @ V))
export class MultiHeadAttention extends Module {
  Wq: Linear;
  Wk: Linear;
  Wv: Linear;
  Wo: Linear;
  private H: number;
  private Dh: number;
  private scale: Tensor;

  constructor(dModel: number, numHeads: number, bias = true) {
    super();
    if (dModel % numHeads !== 0) {
      throw new Error(`MHA: dModel ${dModel} not divisible by numHeads ${numHeads}`);
    }
    this.H = numHeads;
    this.Dh = dModel / numHeads;
    this.Wq = this.child("Wq", new Linear(dModel, dModel, bias));
    this.Wk = this.child("Wk", new Linear(dModel, dModel, bias));
    this.Wv = this.child("Wv", new Linear(dModel, dModel, bias));
    this.Wo = this.child("Wo", new Linear(dModel, dModel, bias));
    this.scale = Tensor.fromData([1 / Math.sqrt(this.Dh)]);
  }

  forward(x: Tensor): Tensor {
    if (x.shape.length !== 3) {
      throw new Error(`MHA input must be [B, S, D], got ${x.shape}`);
    }
    const [B, S, D] = x.shape;
    const H = this.H;
    const Dh = this.Dh;
    if (D !== H * Dh) throw new Error(`MHA dim mismatch: D=${D}, H*Dh=${H * Dh}`);

    // [B, S, D] -> [B, S, H, Dh] -> [B, H, S, Dh] -> [B*H, S, Dh]
    const splitHeads = (t: Tensor): Tensor =>
      t.reshape([B, S, H, Dh]).transpose(1, 2).reshape([B * H, S, Dh]);

    const q = splitHeads(this.Wq.forward(x));
    const k = splitHeads(this.Wk.forward(x));
    const v = splitHeads(this.Wv.forward(x));

    // Scores: [B*H, S, Dh] @ [B*H, Dh, S] -> [B*H, S, S]
    const scores = q.bmm(k.transpose(-1, -2)).mul(this.scale);

    // Softmax over last axis. 2D fast path has max subtraction, so reshape to
    // [B*H*S, S] to hit it.
    const attn = scores.reshape([B * H * S, S]).softmax(-1).reshape([B * H, S, S]);

    // attn @ V: [B*H, S, S] @ [B*H, S, Dh] -> [B*H, S, Dh]
    const context = attn.bmm(v);

    // Merge heads: [B*H, S, Dh] -> [B, H, S, Dh] -> [B, S, H, Dh] -> [B, S, D]
    const merged = context.reshape([B, H, S, Dh]).transpose(1, 2).reshape([B, S, D]);

    return this.Wo.forward(merged);
  }
}

// Pre-norm transformer encoder block: LN → MHA + residual, LN → FFN + residual.
// FFN is Linear(D, 4D) → GELU → Linear(4D, D).
export class TransformerEncoderLayer extends Module {
  ln1: LayerNorm;
  attn: MultiHeadAttention;
  ln2: LayerNorm;
  fc1: Linear;
  fc2: Linear;

  constructor(dModel: number, numHeads: number, ffnMultiplier = 4) {
    super();
    this.ln1 = this.child("ln1", new LayerNorm(dModel));
    this.attn = this.child("attn", new MultiHeadAttention(dModel, numHeads));
    this.ln2 = this.child("ln2", new LayerNorm(dModel));
    this.fc1 = this.child("fc1", new Linear(dModel, dModel * ffnMultiplier));
    this.fc2 = this.child("fc2", new Linear(dModel * ffnMultiplier, dModel));
  }

  forward(x: Tensor): Tensor {
    const a = this.attn.forward(this.ln1.forward(x));
    const r1 = x.add(a);
    const f = this.fc2.forward(this.fc1.forward(this.ln2.forward(r1)).gelu());
    return r1.add(f);
  }
}

// Sinusoidal positional encoding, PyTorch/transformer paper convention.
// Returns a [maxLen, dModel] tensor with requires_grad=false.
export function sinusoidalPositionalEncoding(maxLen: number, dModel: number): Tensor {
  const data = new Float32Array(maxLen * dModel);
  for (let pos = 0; pos < maxLen; pos++) {
    for (let i = 0; i < dModel; i += 2) {
      const angle = pos / Math.pow(10000, i / dModel);
      data[pos * dModel + i] = Math.sin(angle);
      if (i + 1 < dModel) data[pos * dModel + i + 1] = Math.cos(angle);
    }
  }
  return Tensor.fromData(Array.from(data), [maxLen, dModel]);
}
