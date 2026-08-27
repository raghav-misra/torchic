function getOffsets(i: number, shape: number[], stridesA: number[], stridesB: number[]) {
  let idx = i;
  let offsetA = 0;
  let offsetB = 0;
  for (let dim = shape.length - 1; dim >= 0; dim--) {
    const size = shape[dim];
    const pos = idx % size;
    idx = Math.floor(idx / size);
    offsetA += pos * stridesA[dim];
    offsetB += pos * stridesB[dim];
  }
  return [offsetA, offsetB];
}

export function add(
  a: Float32Array,
  b: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  stridesA?: number[],
  stridesB?: number[],
) {
  if (shape && stridesA && stridesB) {
    for (let i = start; i < end; i++) {
      const [offA, offB] = getOffsets(i, shape, stridesA, stridesB);
      out[i] = a[offA] + b[offB];
    }
  } else {
    for (let i = start; i < end; i++) {
      out[i] = a[i] + b[i];
    }
  }
}

export function sub(
  a: Float32Array,
  b: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  stridesA?: number[],
  stridesB?: number[],
) {
  if (shape && stridesA && stridesB) {
    for (let i = start; i < end; i++) {
      const [offA, offB] = getOffsets(i, shape, stridesA, stridesB);
      out[i] = a[offA] - b[offB];
    }
  } else {
    for (let i = start; i < end; i++) {
      out[i] = a[i] - b[i];
    }
  }
}

export function mul(
  a: Float32Array,
  b: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  stridesA?: number[],
  stridesB?: number[],
) {
  if (shape && stridesA && stridesB) {
    for (let i = start; i < end; i++) {
      const [offA, offB] = getOffsets(i, shape, stridesA, stridesB);
      out[i] = a[offA] * b[offB];
    }
  } else {
    for (let i = start; i < end; i++) {
      out[i] = a[i] * b[i];
    }
  }
}

export function div(
  a: Float32Array,
  b: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  stridesA?: number[],
  stridesB?: number[],
) {
  if (shape && stridesA && stridesB) {
    for (let i = start; i < end; i++) {
      const [offA, offB] = getOffsets(i, shape, stridesA, stridesB);
      out[i] = a[offA] / b[offB];
    }
  } else {
    for (let i = start; i < end; i++) {
      out[i] = a[i] / b[i];
    }
  }
}

export function relu(
  a: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) {
      let idx = i;
      let inputOffset = 0;
      for (let dim = shape.length - 1; dim >= 0; dim--) {
        const size = shape[dim];
        const pos = idx % size;
        idx = Math.floor(idx / size);
        inputOffset += pos * strides[dim];
      }
      out[i] = Math.max(0, a[inputOffset]);
    }
  } else {
    for (let i = start; i < end; i++) {
      out[i] = Math.max(0, a[i]);
    }
  }
}

export function exp(
  a: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) {
      let idx = i;
      let inputOffset = 0;
      for (let dim = shape.length - 1; dim >= 0; dim--) {
        const size = shape[dim];
        const pos = idx % size;
        idx = Math.floor(idx / size);
        inputOffset += pos * strides[dim];
      }
      out[i] = Math.exp(a[inputOffset]);
    }
  } else {
    for (let i = start; i < end; i++) {
      out[i] = Math.exp(a[i]);
    }
  }
}

export function log(
  a: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) {
      let idx = i;
      let inputOffset = 0;
      for (let dim = shape.length - 1; dim >= 0; dim--) {
        const size = shape[dim];
        const pos = idx % size;
        idx = Math.floor(idx / size);
        inputOffset += pos * strides[dim];
      }
      out[i] = Math.log(a[inputOffset]);
    }
  } else {
    for (let i = start; i < end; i++) {
      out[i] = Math.log(a[i]);
    }
  }
}

export function fill(out: Float32Array, val: number, start: number, end: number) {
  out.fill(val, start, end);
}

export function randn(out: Float32Array, start: number, end: number) {
  for (let i = start; i < end; i++) {
    // Box-Muller transform
    const u = 1 - Math.random();
    const v = Math.random();
    out[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
}

export function copy(input: Float32Array, out: Float32Array, start: number, end: number) {
  for (let i = start; i < end; i++) {
    out[i] = input[i];
  }
}

export function materialize(
  input: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape: number[],
  strides: number[],
) {
  for (let i = start; i < end; i++) {
    let inputOffset = 0;
    let idx = i;
    for (let dim = shape.length - 1; dim >= 0; dim--) {
      const pos = idx % shape[dim];
      idx = Math.floor(idx / shape[dim]);
      inputOffset += pos * strides[dim];
    }
    out[i] = input[inputOffset];
  }
}

export function relu_backward(
  input: Float32Array,
  gradOutput: Float32Array,
  gradInput: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) {
      let idx = i;
      let inputOffset = 0;
      for (let dim = shape.length - 1; dim >= 0; dim--) {
        const size = shape[dim];
        const pos = idx % size;
        idx = Math.floor(idx / size);
        inputOffset += pos * strides[dim];
      }
      gradInput[i] = input[inputOffset] > 0 ? gradOutput[i] : 0;
    }
  } else {
    for (let i = start; i < end; i++) {
      gradInput[i] = input[i] > 0 ? gradOutput[i] : 0;
    }
  }
}

export function tanh(
  a: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) {
      let idx = i;
      let inputOffset = 0;
      for (let dim = shape.length - 1; dim >= 0; dim--) {
        const size = shape[dim];
        const pos = idx % size;
        idx = Math.floor(idx / size);
        inputOffset += pos * strides[dim];
      }
      out[i] = Math.tanh(a[inputOffset]);
    }
  } else {
    for (let i = start; i < end; i++) {
      out[i] = Math.tanh(a[i]);
    }
  }
}

export function sin(
  a: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) {
      let idx = i;
      let inputOffset = 0;
      for (let dim = shape.length - 1; dim >= 0; dim--) {
        const size = shape[dim];
        const pos = idx % size;
        idx = Math.floor(idx / size);
        inputOffset += pos * strides[dim];
      }
      out[i] = Math.sin(a[inputOffset]);
    }
  } else {
    for (let i = start; i < end; i++) {
      out[i] = Math.sin(a[i]);
    }
  }
}

export function cos(
  a: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) {
      let idx = i;
      let inputOffset = 0;
      for (let dim = shape.length - 1; dim >= 0; dim--) {
        const size = shape[dim];
        const pos = idx % size;
        idx = Math.floor(idx / size);
        inputOffset += pos * strides[dim];
      }
      out[i] = Math.cos(a[inputOffset]);
    }
  } else {
    for (let i = start; i < end; i++) {
      out[i] = Math.cos(a[i]);
    }
  }
}

export function tanh_backward(
  output: Float32Array,
  gradOutput: Float32Array,
  gradInput: Float32Array,
  start: number,
  end: number,
) {
  // derivative of tanh: 1 - output^2
  for (let i = start; i < end; i++) {
    const o = output[i];
    gradInput[i] = gradOutput[i] * (1 - o * o);
  }
}

export function neg(
  a: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) {
      let idx = i;
      let inputOffset = 0;
      for (let dim = shape.length - 1; dim >= 0; dim--) {
        const size = shape[dim];
        const pos = idx % size;
        idx = Math.floor(idx / size);
        inputOffset += pos * strides[dim];
      }
      out[i] = -a[inputOffset];
    }
  } else {
    for (let i = start; i < end; i++) {
      out[i] = -a[i];
    }
  }
}

export function softmax2d(
  input: Float32Array,
  out: Float32Array,
  m: number,
  n: number,
  startRow: number,
  endRow: number,
) {
  for (let r = startRow; r < endRow; r++) {
    const base = r * n;
    let maxv = -Infinity;
    for (let c = 0; c < n; c++) {
      const v = input[base + c];
      if (v > maxv) maxv = v;
    }
    let sum = 0.0;
    for (let c = 0; c < n; c++) {
      const e = Math.exp(input[base + c] - maxv);
      out[base + c] = e;
      sum += e;
    }
    if (sum !== 0) {
      const inv = 1.0 / sum;
      for (let c = 0; c < n; c++) out[base + c] = out[base + c] * inv;
    } else {
      const v = 1.0 / n;
      for (let c = 0; c < n; c++) out[base + c] = v;
    }
  }
}

export function softmax_backward2d(
  output: Float32Array,
  gradOutput: Float32Array,
  gradInput: Float32Array,
  m: number,
  n: number,
  startRow: number,
  endRow: number,
) {
  for (let r = startRow; r < endRow; r++) {
    const base = r * n;
    let dot = 0.0;
    for (let c = 0; c < n; c++) {
      dot += gradOutput[base + c] * output[base + c];
    }
    for (let c = 0; c < n; c++) {
      gradInput[base + c] = output[base + c] * (gradOutput[base + c] - dot);
    }
  }
}

export function rms_norm2d(
  input: Float32Array,
  weight: Float32Array,
  out: Float32Array,
  m: number,
  n: number,
  eps: number,
  startRow: number,
  endRow: number,
) {
  const invN = 1.0 / n;
  for (let r = startRow; r < endRow; r++) {
    const base = r * n;
    let sumsq = 0.0;
    for (let c = 0; c < n; c++) {
      const v = input[base + c];
      sumsq += v * v;
    }
    const invRms = 1.0 / Math.sqrt(sumsq * invN + eps);
    for (let c = 0; c < n; c++) {
      out[base + c] = input[base + c] * invRms * weight[c];
    }
  }
}

export function rope(
  x: Float32Array,
  cos: Float32Array,
  sin: Float32Array,
  out: Float32Array,
  tSeq: number,
  dHalf: number,
  startRow: number,
  endRow: number,
) {
  const d = 2 * dHalf;
  for (let r = startRow; r < endRow; r++) {
    const time = r % tSeq;
    const xBase = r * d;
    const csBase = time * dHalf;
    for (let i = 0; i < dHalf; i++) {
      const a = x[xBase + i];
      const b = x[xBase + i + dHalf];
      const c = cos[csBase + i];
      const s = sin[csBase + i];
      out[xBase + i] = a * c - b * s;
      out[xBase + i + dHalf] = a * s + b * c;
    }
  }
}

export function causal_softmax2d(
  input: Float32Array,
  out: Float32Array,
  m: number,
  n: number,
  pastLen: number,
  startRow: number,
  endRow: number,
) {
  for (let r = startRow; r < endRow; r++) {
    const base = r * n;
    let allowed = pastLen + r;
    if (allowed >= n) allowed = n - 1;
    const endCol = allowed + 1;

    let maxv = -Infinity;
    for (let c = 0; c < endCol; c++) {
      const v = input[base + c];
      if (v > maxv) maxv = v;
    }
    let sum = 0;
    for (let c = 0; c < endCol; c++) {
      const e = Math.exp(input[base + c] - maxv);
      out[base + c] = e;
      sum += e;
    }
    if (sum !== 0) {
      const inv = 1 / sum;
      for (let c = 0; c < endCol; c++) out[base + c] *= inv;
    } else {
      const v = 1 / endCol;
      for (let c = 0; c < endCol; c++) out[base + c] = v;
    }
    for (let c = endCol; c < n; c++) out[base + c] = 0;
  }
}

export function copy_range(
  src: Float32Array,
  dst: Float32Array,
  dstOffset: number,
  start: number,
  end: number,
) {
  for (let i = start; i < end; i++) dst[dstOffset + i] = src[i];
}

export function repeat_interleave(
  input: Float32Array,
  output: Float32Array,
  axisSize: number,
  inner: number,
  repeats: number,
  start: number,
  end: number,
) {
  const strideIn = axisSize * inner;
  const dOut = axisSize * repeats;
  for (let k = start; k < end; k++) {
    const innerIdx = k % inner;
    const rest = (k - innerIdx) / inner;
    const dOutIdx = rest % dOut;
    const o = (rest - dOutIdx) / dOut;
    const dInIdx = (dOutIdx - (dOutIdx % repeats)) / repeats;
    output[k] = input[o * strideIn + dInIdx * inner + innerIdx];
  }
}

// Tanh approximation used by BERT / GPT-2 / Kokoro.
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
const GELU_C = 0.7978845608028654; // sqrt(2/π)
const GELU_B = 0.044715;

function stridedRead(
  a: Float32Array,
  i: number,
  shape: number[],
  strides: number[],
): number {
  let idx = i;
  let off = 0;
  for (let dim = shape.length - 1; dim >= 0; dim--) {
    const size = shape[dim];
    const pos = idx % size;
    idx = Math.floor(idx / size);
    off += pos * strides[dim];
  }
  return a[off];
}

export function gelu(
  a: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) {
      const x = stridedRead(a, i, shape, strides);
      const u = GELU_C * (x + GELU_B * x * x * x);
      out[i] = 0.5 * x * (1 + Math.tanh(u));
    }
  } else {
    for (let i = start; i < end; i++) {
      const x = a[i];
      const u = GELU_C * (x + GELU_B * x * x * x);
      out[i] = 0.5 * x * (1 + Math.tanh(u));
    }
  }
}

export function gelu_backward(
  input: Float32Array,
  gradOutput: Float32Array,
  gradInput: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) {
      const x = stridedRead(input, i, shape, strides);
      const x2 = x * x;
      const u = GELU_C * (x + GELU_B * x * x2);
      const t = Math.tanh(u);
      const dudx = GELU_C * (1 + 3 * GELU_B * x2);
      const dgelu = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dudx;
      gradInput[i] = gradOutput[i] * dgelu;
    }
  } else {
    for (let i = start; i < end; i++) {
      const x = input[i];
      const x2 = x * x;
      const u = GELU_C * (x + GELU_B * x * x2);
      const t = Math.tanh(u);
      const dudx = GELU_C * (1 + 3 * GELU_B * x2);
      const dgelu = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dudx;
      gradInput[i] = gradOutput[i] * dgelu;
    }
  }
}

export function sqrt(
  a: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) out[i] = Math.sqrt(stridedRead(a, i, shape, strides));
  } else {
    for (let i = start; i < end; i++) out[i] = Math.sqrt(a[i]);
  }
}

// d/dx sqrt(x) = 0.5 / sqrt(x) = 0.5 / y
export function sqrt_backward(
  output: Float32Array,
  gradOutput: Float32Array,
  gradInput: Float32Array,
  start: number,
  end: number,
) {
  for (let i = start; i < end; i++) {
    gradInput[i] = gradOutput[i] * 0.5 / output[i];
  }
}

export function rsqrt(
  a: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) out[i] = 1 / Math.sqrt(stridedRead(a, i, shape, strides));
  } else {
    for (let i = start; i < end; i++) out[i] = 1 / Math.sqrt(a[i]);
  }
}

// d/dx x^(-1/2) = -0.5 * x^(-3/2) = -0.5 * y^3
export function rsqrt_backward(
  output: Float32Array,
  gradOutput: Float32Array,
  gradInput: Float32Array,
  start: number,
  end: number,
) {
  for (let i = start; i < end; i++) {
    const y = output[i];
    gradInput[i] = gradOutput[i] * -0.5 * y * y * y;
  }
}

// sigmoid(x) = 1 / (1 + exp(-x))
export function sigmoid(
  a: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) {
      const x = stridedRead(a, i, shape, strides);
      out[i] = 1 / (1 + Math.exp(-x));
    }
  } else {
    for (let i = start; i < end; i++) out[i] = 1 / (1 + Math.exp(-a[i]));
  }
}

// d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) = y * (1 - y)
export function sigmoid_backward(
  output: Float32Array,
  gradOutput: Float32Array,
  gradInput: Float32Array,
  start: number,
  end: number,
) {
  for (let i = start; i < end; i++) {
    const y = output[i];
    gradInput[i] = gradOutput[i] * y * (1 - y);
  }
}

export function leaky_relu(
  a: Float32Array,
  out: Float32Array,
  negativeSlope: number,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) {
      const x = stridedRead(a, i, shape, strides);
      out[i] = x >= 0 ? x : x * negativeSlope;
    }
  } else {
    for (let i = start; i < end; i++) {
      const x = a[i];
      out[i] = x >= 0 ? x : x * negativeSlope;
    }
  }
}

export function leaky_relu_backward(
  input: Float32Array,
  gradOutput: Float32Array,
  gradInput: Float32Array,
  negativeSlope: number,
  start: number,
  end: number,
) {
  for (let i = start; i < end; i++) {
    const x = input[i];
    gradInput[i] = gradOutput[i] * (x >= 0 ? 1 : negativeSlope);
  }
}

// SiLU / Swish: y = x * sigmoid(x)
export function silu(
  a: Float32Array,
  out: Float32Array,
  start: number,
  end: number,
  shape?: number[],
  strides?: number[],
) {
  if (shape && strides) {
    for (let i = start; i < end; i++) {
      const x = stridedRead(a, i, shape, strides);
      out[i] = x / (1 + Math.exp(-x));
    }
  } else {
    for (let i = start; i < end; i++) {
      const x = a[i];
      out[i] = x / (1 + Math.exp(-x));
    }
  }
}

// d/dx (x * sigmoid(x)) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
export function silu_backward(
  input: Float32Array,
  gradOutput: Float32Array,
  gradInput: Float32Array,
  start: number,
  end: number,
) {
  for (let i = start; i < end; i++) {
    const x = input[i];
    const s = 1 / (1 + Math.exp(-x));
    gradInput[i] = gradOutput[i] * s * (1 + x * (1 - s));
  }
}
