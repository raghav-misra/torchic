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

// GroupNorm over a [B, C, ...] tensor. Splits C into `numGroups`, computes
// mean/var over (channels-per-group * spatial dims), applies per-channel
// affine. Matches PyTorch semantics; composed from primitives.
export class GroupNorm extends Module {
  weight: Tensor;
  bias: Tensor;
  private eps: Tensor;
  private numGroups: number;
  private numChannels: number;

  constructor(numGroups: number, numChannels: number, eps = 1e-5) {
    super();
    if (numChannels % numGroups !== 0) {
      throw new Error(`GroupNorm: channels ${numChannels} not divisible by groups ${numGroups}`);
    }
    this.numGroups = numGroups;
    this.numChannels = numChannels;
    this.weight = this.param("weight", Tensor.ones([numChannels], true));
    this.bias = this.param("bias", Tensor.zeros([numChannels], true));
    this.eps = Tensor.fromData([eps]);
  }

  forward(x: Tensor): Tensor {
    const shape = x.shape;
    if (shape.length < 2) throw new Error(`GroupNorm input must be at least [B, C], got ${shape}`);
    const B = shape[0];
    const C = shape[1];
    if (C !== this.numChannels) {
      throw new Error(`GroupNorm expected C=${this.numChannels}, got ${C}`);
    }
    const G = this.numGroups;
    const Cg = C / G;
    let spatial = 1;
    for (let i = 2; i < shape.length; i++) spatial *= shape[i];

    // Normalize over (Cg * spatial). Reshape to [B, G, Cg*spatial], mean-var
    // along last axis, then reshape back.
    const grouped = x.reshape([B, G, Cg * spatial]);
    const mean = grouped.mean(-1, true);
    const centered = grouped.sub(mean);
    const variance = centered.mul(centered).mean(-1, true);
    const invStd = variance.add(this.eps).rsqrt();
    const norm = centered.mul(invStd).reshape(shape);

    // Broadcast per-channel affine. weight/bias are [C]; reshape to [1, C, 1, ...]
    // so they broadcast across batch and spatial.
    const affineShape = new Array(shape.length).fill(1);
    affineShape[1] = C;
    const w = this.weight.reshape(affineShape);
    const b = this.bias.reshape(affineShape);
    return norm.mul(w).add(b);
  }
}

// InstanceNorm over [B, C, ...] is GroupNorm with G=C: each channel of each
// sample is independently normalized. Common in StyleTTS2 decoder blocks.
export class InstanceNorm1d extends Module {
  weight: Tensor | null;
  bias: Tensor | null;
  private eps: Tensor;
  private affine: boolean;
  private numFeatures: number;

  constructor(numFeatures: number, eps = 1e-5, affine = false) {
    super();
    this.numFeatures = numFeatures;
    this.affine = affine;
    this.eps = Tensor.fromData([eps]);
    if (affine) {
      this.weight = this.param("weight", Tensor.ones([numFeatures], true));
      this.bias = this.param("bias", Tensor.zeros([numFeatures], true));
    } else {
      this.weight = null;
      this.bias = null;
    }
  }

  forward(x: Tensor): Tensor {
    if (x.shape.length !== 3) throw new Error(`InstanceNorm1d input must be [B, C, L], got ${x.shape}`);
    const [_B, C, _L] = x.shape;
    if (C !== this.numFeatures) throw new Error(`InstanceNorm1d expected C=${this.numFeatures}, got ${C}`);
    // Normalize over the last dim (L). Mean/var per (B, C, :).
    const mean = x.mean(-1, true);
    const centered = x.sub(mean);
    const variance = centered.mul(centered).mean(-1, true);
    const invStd = variance.add(this.eps).rsqrt();
    const norm = centered.mul(invStd);
    if (this.affine && this.weight && this.bias) {
      const w = this.weight.reshape([1, C, 1]);
      const b = this.bias.reshape([1, C, 1]);
      return norm.mul(w).add(b);
    }
    return norm;
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

interface Conv1dOptions {
  stride?: number;
  padding?: number;
  dilation?: number;
  bias?: boolean;
}

export class Conv1d extends Module {
  weight: Tensor;
  bias: Tensor | null;
  private stride: number;
  private padding: number;
  private dilation: number;

  constructor(inChannels: number, outChannels: number, kernelSize: number, opts: Conv1dOptions = {}) {
    super();
    this.stride = opts.stride ?? 1;
    this.padding = opts.padding ?? 0;
    this.dilation = opts.dilation ?? 1;
    const fanIn = inChannels * kernelSize;
    this.weight = this.param(
      "weight",
      scaledRandn([outChannels, inChannels, kernelSize], Math.sqrt(2 / fanIn)),
    );
    this.bias = (opts.bias ?? true) ? this.param("bias", Tensor.zeros([outChannels], true)) : null;
  }

  forward(x: Tensor): Tensor {
    return x.conv1d(this.weight, this.bias, {
      stride: this.stride,
      padding: this.padding,
      dilation: this.dilation,
    });
  }
}

interface ConvTranspose1dOptions extends Conv1dOptions {
  outputPadding?: number;
}

export class ConvTranspose1d extends Module {
  weight: Tensor;
  bias: Tensor | null;
  private stride: number;
  private padding: number;
  private dilation: number;
  private outputPadding: number;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    opts: ConvTranspose1dOptions = {},
  ) {
    super();
    this.stride = opts.stride ?? 1;
    this.padding = opts.padding ?? 0;
    this.dilation = opts.dilation ?? 1;
    this.outputPadding = opts.outputPadding ?? 0;
    const fanIn = outChannels * kernelSize;
    this.weight = this.param(
      "weight",
      scaledRandn([inChannels, outChannels, kernelSize], Math.sqrt(2 / fanIn)),
    );
    this.bias = (opts.bias ?? true) ? this.param("bias", Tensor.zeros([outChannels], true)) : null;
  }

  forward(x: Tensor): Tensor {
    return x.convTranspose1d(this.weight, this.bias, {
      stride: this.stride,
      padding: this.padding,
      dilation: this.dilation,
      outputPadding: this.outputPadding,
    });
  }
}

// Standard PyTorch-layout LSTM cell. weight_ih is [4*hidden, input] (concat of
// i,f,g,o gate weights along output dim). Compatible with HuggingFace/PyTorch
// safetensors naming so a Kokoro state_dict loads with no key remap.
export class LSTMCell extends Module {
  weight_ih: Tensor;
  weight_hh: Tensor;
  bias_ih: Tensor;
  bias_hh: Tensor;
  private hidden: number;

  constructor(inputSize: number, hiddenSize: number) {
    super();
    this.hidden = hiddenSize;
    // PyTorch default: uniform in [-1/sqrt(H), 1/sqrt(H)] — we approximate with
    // the same stddev via normal init.
    const k = 1 / Math.sqrt(hiddenSize);
    this.weight_ih = this.param("weight_ih", scaledRandn([4 * hiddenSize, inputSize], k));
    this.weight_hh = this.param("weight_hh", scaledRandn([4 * hiddenSize, hiddenSize], k));
    this.bias_ih = this.param("bias_ih", Tensor.zeros([4 * hiddenSize], true));
    this.bias_hh = this.param("bias_hh", Tensor.zeros([4 * hiddenSize], true));
  }

  // x: [B, input_size], (h, c): each [B, hidden_size] -> new (h, c) each [B, hidden_size]
  forward(x: Tensor, state: [Tensor, Tensor]): [Tensor, Tensor] {
    const [h, c] = state;
    if (x.shape.length !== 2) throw new Error(`LSTMCell input must be [B, in], got ${x.shape}`);
    const B = x.shape[0];
    const H = this.hidden;

    const gates = x
      .matmul(this.weight_ih.transpose())
      .add(this.bias_ih)
      .add(h.matmul(this.weight_hh.transpose()))
      .add(this.bias_hh);

    const iGate = gates.slice([[0, B], [0, H]]).sigmoid();
    const fGate = gates.slice([[0, B], [H, 2 * H]]).sigmoid();
    const gCand = gates.slice([[0, B], [2 * H, 3 * H]]).tanh();
    const oGate = gates.slice([[0, B], [3 * H, 4 * H]]).sigmoid();

    const cNew = fGate.mul(c).add(iGate.mul(gCand));
    const hNew = oGate.mul(cNew.tanh());
    return [hNew, cNew];
  }

  // Zero-initialized (h, c). Detached from any graph.
  initialState(batchSize: number): [Tensor, Tensor] {
    return [Tensor.zeros([batchSize, this.hidden]), Tensor.zeros([batchSize, this.hidden])];
  }
}

// Helper: unroll an LSTMCell over the time dim (batch-first). Returns the
// stacked hidden states [B, T, H] and the final (h, c). No dropout, no
// state passing between calls — this is for inference.
export function lstmForward(
  cell: LSTMCell,
  x: Tensor,
  init?: [Tensor, Tensor],
): { output: Tensor; hFinal: Tensor; cFinal: Tensor } {
  if (x.shape.length !== 3) throw new Error(`lstmForward input must be [B, T, in], got ${x.shape}`);
  const [B, T, _in] = x.shape;
  const [h0, c0] = init ?? cell.initialState(B);
  const H = h0.shape[1];
  let h = h0;
  let c = c0;
  const outputs: Tensor[] = [];
  for (let t = 0; t < T; t++) {
    const xt = x.slice([[0, B], [t, t + 1], [0, x.shape[2]]]).reshape([B, x.shape[2]]);
    [h, c] = cell.forward(xt, [h, c]);
    outputs.push(h.reshape([B, 1, H]));
  }
  const output = Tensor.concat(outputs, 1);
  return { output, hFinal: h, cFinal: c };
}

// Reverse a tensor along the time axis (batch-first: axis=1). Uses slice+concat.
function reverseTime(x: Tensor): Tensor {
  const T = x.shape[1];
  const chunks: Tensor[] = [];
  for (let t = T - 1; t >= 0; t--) {
    chunks.push(x.slice([[0, x.shape[0]], [t, t + 1], [0, x.shape[2]]]));
  }
  return Tensor.concat(chunks, 1);
}

// Bidirectional LSTM. Composed from two LSTMCell instances (forward + reverse
// unroll). Output is [B, T, 2*hidden]. State dict keys match PyTorch:
// weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0 (forward)
// weight_ih_l0_reverse, weight_hh_l0_reverse, bias_ih_l0_reverse, bias_hh_l0_reverse
export class BiLSTM extends Module {
  fwd: LSTMCell;
  bwd: LSTMCell;
  private hidden: number;

  constructor(inputSize: number, hiddenSize: number) {
    super();
    this.hidden = hiddenSize;
    this.fwd = this.child("fwd", new LSTMCell(inputSize, hiddenSize));
    this.bwd = this.child("bwd", new LSTMCell(inputSize, hiddenSize));
  }

  forward(x: Tensor): Tensor {
    const fwdOut = lstmForward(this.fwd, x).output;
    const rev = reverseTime(x);
    const bwdOut = reverseTime(lstmForward(this.bwd, rev).output);
    return Tensor.concat([fwdOut, bwdOut], -1);
  }
}
