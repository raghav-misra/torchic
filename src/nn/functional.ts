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

export function causal_softmax(x: Tensor, pastLen = 0): Tensor {
  return x.causal_softmax(pastLen);
}
