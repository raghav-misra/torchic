import { Tensor } from "../frontend/tensor";

// Stateless activations. Layers-with-state (Linear, LayerNorm, ...) live in ./layers.

export function relu(x: Tensor): Tensor {
  return x.relu();
}

export function tanh(x: Tensor): Tensor {
  return x.tanh();
}

export function sin(x: Tensor): Tensor {
  return x.sin();
}

export function cos(x: Tensor): Tensor {
  return x.cos();
}

export function gelu(x: Tensor): Tensor {
  return x.gelu();
}

export function sigmoid(x: Tensor): Tensor {
  return x.sigmoid();
}

export function leaky_relu(x: Tensor, negativeSlope = 0.01): Tensor {
  return x.leaky_relu(negativeSlope);
}

export function silu(x: Tensor): Tensor {
  return x.silu();
}

export function rms_norm(x: Tensor, weight: Tensor, eps = 1e-5): Tensor {
  return x.rms_norm(weight, eps);
}

export function rope(x: Tensor, cos: Tensor, sin: Tensor): Tensor {
  return x.rope(cos, sin);
}

// Precompute RoPE cos/sin tables. Returns [maxSeqLen, headDim/2] tensors that
// slot directly into rope(). theta defaults to 10000; Llama 3.2 uses 500000.
export function precomputeRope(
  maxSeqLen: number,
  headDim: number,
  theta = 10000,
): { cos: Tensor; sin: Tensor } {
  if (headDim % 2 !== 0) {
    throw new Error(`precomputeRope: headDim must be even, got ${headDim}`);
  }
  const dHalf = headDim / 2;
  const cosData = new Float32Array(maxSeqLen * dHalf);
  const sinData = new Float32Array(maxSeqLen * dHalf);
  for (let i = 0; i < dHalf; i++) {
    const invFreq = Math.pow(theta, (-2 * i) / headDim);
    for (let t = 0; t < maxSeqLen; t++) {
      const angle = t * invFreq;
      cosData[t * dHalf + i] = Math.cos(angle);
      sinData[t * dHalf + i] = Math.sin(angle);
    }
  }
  return {
    cos: Tensor.fromData(cosData, [maxSeqLen, dHalf]),
    sin: Tensor.fromData(sinData, [maxSeqLen, dHalf]),
  };
}

export function softmax(x: Tensor, axis = -1): Tensor {
  return x.softmax(axis);
}

export function causal_softmax(x: Tensor, pastLen = 0, tQuery?: number): Tensor {
  return x.causal_softmax(pastLen, tQuery);
}

// Causal scaled dot-product attention. q [N, Tq, D], k/v [N, Tk, D] where
// Tk = pastLen + Tq. Returns [N, Tq, D]. N is typically B*H (caller reshapes
// from [B, H, T, D] before calling). For GQA, caller should already have
// broadcast K/V heads to match H via .repeatInterleave. Inference-only.
export function causalAttention(q: Tensor, k: Tensor, v: Tensor, pastLen = 0): Tensor {
  if (q.shape.length !== 3) throw new Error(`causalAttention: q must be 3D [N, Tq, D], got ${q.shape}`);
  if (k.shape.length !== 3) throw new Error(`causalAttention: k must be 3D [N, Tk, D], got ${k.shape}`);
  if (v.shape.length !== 3) throw new Error(`causalAttention: v must be 3D [N, Tk, D], got ${v.shape}`);
  const [N, Tq, D] = q.shape;
  const [Nk, Tk, Dk] = k.shape;
  const [Nv, Tkv, Dv] = v.shape;
  if (N !== Nk || N !== Nv) {
    throw new Error(`causalAttention: batch dim mismatch q=${N} k=${Nk} v=${Nv}`);
  }
  if (D !== Dk || D !== Dv) {
    throw new Error(`causalAttention: head dim mismatch q=${D} k=${Dk} v=${Dv}`);
  }
  if (Tk !== Tkv) {
    throw new Error(`causalAttention: k/v seq mismatch ${Tk} vs ${Tkv}`);
  }
  if (Tk !== pastLen + Tq) {
    throw new Error(`causalAttention: expected Tk = pastLen + Tq = ${pastLen + Tq}, got ${Tk}`);
  }

  const scale = Tensor.fromData([1 / Math.sqrt(D)]);
  const kT = k.transpose(-1, -2);
  const scoresRaw = q.bmm(kT);
  kT.dispose();
  const scores = scoresRaw.mul(scale);
  scoresRaw.dispose();
  scale.dispose();

  const attnFlat = scores.reshape([N * Tq, Tk]).causal_softmax(pastLen, Tq);
  scores.dispose();
  const attn = attnFlat.reshape([N, Tq, Tk]);
  attnFlat.dispose();

  const out = attn.bmm(v);
  attn.dispose();
  return out;
}
