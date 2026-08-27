import { getDispatcher, isDispatcherReady } from "./dispatcher";
import type { OpParams } from "../shared/types";
export type NestedArray = number | NestedArray[];

export function inferShape(arr: NestedArray | Float32Array): number[] {
  if (arr instanceof Float32Array) return [arr.length];
  if (!Array.isArray(arr)) return [];
  const dims: number[] = [];
  let curr: NestedArray = arr;
  while (Array.isArray(curr)) {
    dims.push(curr.length);
    curr = curr[0];
  }
  return dims;
}

export function countElements(arr: NestedArray): number {
  if (!Array.isArray(arr)) return 1;
  let total = 0;
  for (const el of arr) total += countElements(el);
  return total;
}

export function flattenInto(arr: NestedArray, out: Float32Array, offset: number): number {
  if (!Array.isArray(arr)) {
    out[offset] = arr;
    return offset + 1;
  }
  for (const el of arr) offset = flattenInto(el, out, offset);
  return offset;
}

// GC may fire after shutdown; skip freeing if there's no dispatcher.
const registry = new FinalizationRegistry((id: string) => {
  if (!isDispatcherReady()) return;
  getDispatcher().free(id);
});

export const GradMode = {
  enabled: true,
};

let _activeTracking: Set<Tensor> | null = null;
let _backwardTracking: Set<Tensor> | null = null;

export async function noGrad<T>(fn: () => Promise<T>): Promise<T> {
  const prev = GradMode.enabled;
  GradMode.enabled = false;
  return fn().finally(() => {
    GradMode.enabled = prev;
  });
}

export function noGradSync<T>(fn: () => T): T {
  const prev = GradMode.enabled;
  GradMode.enabled = false;
  try {
    return fn();
  } finally {
    GradMode.enabled = prev;
  }
}

export async function trackTensors<T>(fn: () => Promise<T>): Promise<T> {
  if (_activeTracking) {
    throw new Error("Nested tracking not supported yet");
  }
  _activeTracking = new Set();
  try {
    return await fn();
  } finally {
    const tracked = _activeTracking;
    _activeTracking = null;
    if (tracked) {
      for (const t of tracked) {
        t.dispose();
      }
    }
  }
}

export type { OpParams } from "../shared/types";

export class Tensor {
  id: string;
  shape: number[];
  strides: number[];
  offset: number; // Byte offset in SharedArrayBuffer
  requiresGrad: boolean;
  grad: Tensor | null = null;

  op: string | null = null;
  prev: Tensor[] = [];
  params: OpParams = {};
  isDisposed = false;

  constructor(id: string, shape: number[], requiresGrad = false, offset = 0, strides?: number[]) {
    this.id = id;
    this.shape = shape;
    this.strides = strides ? strides.slice() : Tensor.computeStrides(shape);
    this.offset = offset;
    this.requiresGrad = requiresGrad;

    registry.register(this, this.id, this);

    if (_activeTracking) {
      _activeTracking.add(this);
    }
    if (_backwardTracking) {
      _backwardTracking.add(this);
    }
  }
  /**
   * Returns a zero-copy n-dimensional slice view of this tensor.
   * @param ranges Array of [start, end) for each dimension
   */
  slice(ranges: [number, number][]): Tensor {
    if (ranges.length !== this.shape.length) {
      throw new Error(
        `slice: ranges length ${ranges.length} does not match tensor rank ${this.shape.length}`,
      );
    }
    const newShape = ranges.map(([start, end], i) => {
      if (start < 0 || end > this.shape[i] || start >= end) {
        throw new Error(
          `Invalid slice range [${start}, ${end}) for dimension ${i} with size ${this.shape[i]}`,
        );
      }
      return end - start;
    });
    const relativeStartElements = ranges.reduce(
      (acc, [start], i) => acc + start * this.strides[i],
      0,
    );

    const viewId = getDispatcher().nextTensorId();
    const relativeOffsetBytes = relativeStartElements * 4;
    getDispatcher().allocateView(viewId, this.id, relativeOffsetBytes);

    const newOffset = this.offset + relativeOffsetBytes;
    const out = new Tensor(viewId, newShape, this.requiresGrad, newOffset, this.strides);
    // Views share the parent's buffer. Hold a strong ref so GC can't reclaim
    // the parent (via FinalizationRegistry) while the view is still live.
    out.prev = [this];
    return out;
  }

  /**
   * Sets the value at the given n-dimensional indices.
   * @param indices Array of indices for each dimension
   * @param value Value to set
   */
  set(indices: number[], value: number) {
    if (indices.length !== this.shape.length) {
      throw new Error(
        `set: indices length ${indices.length} does not match tensor rank ${this.shape.length}`,
      );
    }
    let flatIndex = 0;
    for (let i = 0; i < indices.length; i++) {
      if (indices[i] < 0 || indices[i] >= this.shape[i]) {
        throw new Error(
          `set: index ${indices[i]} out of bounds for dimension ${i} with size ${this.shape[i]}`,
        );
      }
      flatIndex += indices[i] * this.strides[i];
    }
    // offset is in bytes, strides are in elements
    flatIndex += this.offset / 4;
    getDispatcher().set(this.id, flatIndex, value);
  }

  dispose() {
    if (this.isDisposed) return;
    this.isDisposed = true;
    registry.unregister(this);
    getDispatcher().free(this.id);
  }

  private static computeStrides(shape: number[]): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  private isContiguous(): boolean {
    // Size-1 dims have exactly one element, so their stride is a "don't care"
    // for row-major layout — a slice that gives shape [1, 1, N] over a [B, T, N]
    // source is still N contiguous elements regardless of the outer strides.
    let expectedStride = 1;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      if (this.shape[i] === 1) continue;
      if (this.strides[i] !== expectedStride) return false;
      expectedStride *= this.shape[i];
    }
    return true;
  }

  private materialize(): Tensor {
    if (this.isContiguous()) return this;

    const outId = getDispatcher().nextTensorId();
    const size = this.numElements() * 4;

    getDispatcher().allocate(outId, size);
    getDispatcher().runOp("MATERIALIZE", [this.id], outId, {
      shape: this.shape,
      strides: this.strides,
    });

    const out = new Tensor(outId, this.shape, this.requiresGrad);
    if (this.requiresGrad && GradMode.enabled) {
      out.op = "MATERIALIZE";
    }
    // Hold the source alive until the queued MATERIALIZE op has actually run.
    // Also protects downstream reshape views that end up viewing `out`.
    out.prev = [this];
    return out;
  }

  static ones(shape: number[], requiresGrad = false): Tensor {
    return Tensor.create(shape, requiresGrad, "FILL", { value: 1 });
  }

  static zeros(shape: number[], requiresGrad = false): Tensor {
    return Tensor.create(shape, requiresGrad, "FILL", { value: 0 });
  }

  static randn(shape: number[], requiresGrad = false): Tensor {
    return Tensor.create(shape, requiresGrad, "RANDN");
  }

  /**
   * Creates a tensor from nested arrays or flat data. Infers shape if not provided.
   * @param data Nested array (any depth) or flat array/Float32Array
   * @param shape Optional shape. If not provided, inferred from data.
   * @param requiresGrad Whether to track gradients
   */
  static fromData(
    data: NestedArray | Float32Array,
    shape?: number[],
    requiresGrad = false,
  ): Tensor {
    const finalShape = shape ?? inferShape(data);

    let typedData: Float32Array;
    if (data instanceof Float32Array) {
      typedData = data;
    } else {
      const n = countElements(data);
      typedData = new Float32Array(n);
      flattenInto(data, typedData, 0);
    }

    const size = finalShape.reduce((a, b) => a * b, 1) * 4;
    const id = getDispatcher().nextTensorId();

    getDispatcher().allocate(id, size);
    getDispatcher().write(id, typedData);

    return new Tensor(id, finalShape, requiresGrad);
  }

  /**
   * Allocate an uninitialized tensor buffer of the given shape.
   * Useful for reusing the same backing memory across iterations.
   */
  static empty(shape: number[], requiresGrad = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1) * 4;
    const id = getDispatcher().nextTensorId();
    getDispatcher().allocate(id, size);
    return new Tensor(id, shape, requiresGrad);
  }

  // Concatenate tensors along `axis`. All non-axis dims must match.
  // Inference-only: no autograd wiring on the output.
  static concat(tensors: Tensor[], axis: number): Tensor {
    if (tensors.length === 0) throw new Error(`concat: no tensors`);
    const first = tensors[0];
    const rank = first.shape.length;
    const normAxis = axis < 0 ? axis + rank : axis;
    if (normAxis < 0 || normAxis >= rank) throw new Error(`concat: bad axis ${axis} for rank ${rank}`);
    let outAxisSize = 0;
    for (const t of tensors) {
      if (t.shape.length !== rank) throw new Error(`concat: rank mismatch ${first.shape} vs ${t.shape}`);
      for (let d = 0; d < rank; d++) {
        if (d === normAxis) continue;
        if (t.shape[d] !== first.shape[d]) {
          throw new Error(`concat: dim ${d} mismatch ${first.shape} vs ${t.shape}`);
        }
      }
      outAxisSize += t.shape[normAxis];
    }
    let outerSize = 1;
    for (let d = 0; d < normAxis; d++) outerSize *= first.shape[d];
    let innerSize = 1;
    for (let d = normAxis + 1; d < rank; d++) innerSize *= first.shape[d];

    const outShape = first.shape.slice();
    outShape[normAxis] = outAxisSize;
    const outId = getDispatcher().nextTensorId();
    getDispatcher().allocate(outId, outShape.reduce((a, b) => a * b, 1) * 4);

    let axisOffset = 0;
    for (const t of tensors) {
      const src = t.materialize();
      const inAxisSize = t.shape[normAxis];
      getDispatcher().runOp("CONCAT_SLAB", [src.id], outId, {
        outerSize,
        inAxisSize,
        outAxisSize,
        axisOffset,
        innerSize,
      });
      if (src !== t) src.dispose();
      axisOffset += inAxisSize;
    }
    return new Tensor(outId, outShape, false);
  }

  // Constant padding on the last dim. Reflect/replicate modes will need
  // their own kernels; concat + fill is only correct for a constant value.
  pad1d(left: number, right: number, value = 0): Tensor {
    if (left < 0 || right < 0) throw new Error(`pad1d: negative pad`);
    if (left === 0 && right === 0) return this;
    const shape = this.shape.slice();
    const rank = shape.length;
    if (rank < 1) throw new Error(`pad1d: needs at least 1D input`);
    const parts: Tensor[] = [];
    if (left > 0) {
      const pShape = shape.slice();
      pShape[rank - 1] = left;
      const p = Tensor.zeros(pShape);
      if (value !== 0) {
        const size = pShape.reduce((a, b) => a * b, 1);
        p.write(new Float32Array(size).fill(value));
      }
      parts.push(p);
    }
    parts.push(this);
    if (right > 0) {
      const pShape = shape.slice();
      pShape[rank - 1] = right;
      const p = Tensor.zeros(pShape);
      if (value !== 0) {
        const size = pShape.reduce((a, b) => a * b, 1);
        p.write(new Float32Array(size).fill(value));
      }
      parts.push(p);
    }
    return Tensor.concat(parts, -1);
  }

  // Reflection padding on the last dim. Composed from slice + concat (no kernel).
  // Matches nn.ReflectionPad1d: boundary elements are the reflection axis.
  reflectionPad1d(left: number, right: number): Tensor {
    if (left < 0 || right < 0) throw new Error(`reflectionPad1d: negative pad`);
    if (left === 0 && right === 0) return this;
    const rank = this.shape.length;
    if (rank < 1) throw new Error(`reflectionPad1d: needs at least 1D input`);
    const L = this.shape[rank - 1];
    if (left >= L || right >= L) {
      throw new Error(`reflectionPad1d: pad(${left},${right}) must be < last dim ${L}`);
    }
    const buildRange = (start: number, end: number): [number, number][] => {
      const rs: [number, number][] = [];
      for (let d = 0; d < rank; d++) {
        rs.push(d === rank - 1 ? [start, end] : [0, this.shape[d]]);
      }
      return rs;
    };
    const parts: Tensor[] = [];
    for (let i = left; i >= 1; i--) parts.push(this.slice(buildRange(i, i + 1)));
    parts.push(this);
    for (let i = 1; i <= right; i++) parts.push(this.slice(buildRange(L - 1 - i, L - i)));
    return Tensor.concat(parts, -1);
  }

  // Split into `sections` equal-sized chunks along `axis` (last dim by default).
  // Zero-copy: each chunk is a slice view. Requires shape[axis] divisible by sections.
  split(sections: number, axis = -1): Tensor[] {
    const rank = this.shape.length;
    const normAxis = axis < 0 ? axis + rank : axis;
    const dim = this.shape[normAxis];
    if (dim % sections !== 0) throw new Error(`split: ${dim} not divisible by ${sections}`);
    const chunk = dim / sections;
    const out: Tensor[] = [];
    for (let s = 0; s < sections; s++) {
      const ranges: [number, number][] = this.shape.map((d, i) =>
        i === normAxis ? [s * chunk, (s + 1) * chunk] as [number, number] : [0, d],
      );
      out.push(this.slice(ranges));
    }
    return out;
  }

  /**
   * Write data into this tensor's backing buffer. `data` should be a
   * Float32Array (or array convertible to it); if shorter than the
   * tensor's storage it will overwrite the prefix.
   */
  write(data: Float32Array | number[]) {
    const arr = data instanceof Float32Array ? data : new Float32Array(data);
    getDispatcher().write(this.id, arr);
  }

  private static create(
    shape: number[],
    requiresGrad: boolean,
    op: string,
    params: OpParams = {},
  ): Tensor {
    const size = shape.reduce((a, b) => a * b, 1) * 4; // 4 bytes per float
    const id = getDispatcher().nextTensorId();

    getDispatcher().allocate(id, size);
    getDispatcher().runOp(op, [], id, params);

    return new Tensor(id, shape, requiresGrad);
  }

  add(other: Tensor): Tensor {
    return this.runBinaryOp("ADD", other);
  }

  sub(other: Tensor): Tensor {
    return this.runBinaryOp("SUB", other);
  }

  mul(other: Tensor): Tensor {
    return this.runBinaryOp("MUL", other);
  }

  div(other: Tensor): Tensor {
    return this.runBinaryOp("DIV", other);
  }

  relu(): Tensor {
    return this.runUnaryOp("RELU");
  }

  exp(): Tensor {
    return this.runUnaryOp("EXP");
  }

  log(): Tensor {
    return this.runUnaryOp("LOG");
  }

  tanh(): Tensor {
    return this.runUnaryOp("TANH");
  }

  sin(): Tensor {
    return this.runUnaryOp("SIN");
  }

  cos(): Tensor {
    return this.runUnaryOp("COS");
  }

  // Fused Snake activation: y = x + (1/α)·sin²(α·x). α broadcasts per channel.
  // Expects `this` shape [B, C, T] contiguous and `alpha` with numel == C.
  snake(alpha: Tensor): Tensor {
    if (this.shape.length !== 3) {
      throw new Error(`snake: expected [B, C, T] input, got shape [${this.shape.join(",")}]`);
    }
    const [, C, T] = this.shape;
    const alphaNumel = alpha.shape.reduce((a, b) => a * b, 1);
    if (alphaNumel !== C) {
      throw new Error(`snake: alpha numel ${alphaNumel} must equal C=${C}`);
    }
    const outId = getDispatcher().nextTensorId();
    getDispatcher().allocate(outId, this.numElements() * 4);
    getDispatcher().runOp("SNAKE_1D", [this.id, alpha.id], outId, {
      axisSize: C,
      innerSize: T,
    });
    return new Tensor(outId, this.shape.slice(), false);
  }

  // Fused StyleTTS AdaIN affine: y = x * (1 + gamma) + beta, gamma/beta
  // broadcast per channel. `axis` names the channel axis of x (1 for [B,C,L],
  // -1 or 2 for [B,T,C]). Requires gamma/beta numel == C (i.e. per-channel,
  // broadcast across all other dims).
  styleAffine(gamma: Tensor, beta: Tensor, axis: number): Tensor {
    const rank = this.shape.length;
    const ax = axis < 0 ? rank + axis : axis;
    if (ax < 0 || ax >= rank) throw new Error(`styleAffine: axis ${axis} out of range for rank ${rank}`);
    const C = this.shape[ax];
    const inner = this.shape.slice(ax + 1).reduce((a, b) => a * b, 1);
    const gammaNumel = gamma.shape.reduce((a, b) => a * b, 1);
    const betaNumel = beta.shape.reduce((a, b) => a * b, 1);
    if (gammaNumel !== C || betaNumel !== C) {
      throw new Error(`styleAffine: gamma/beta numel must equal C=${C}, got ${gammaNumel}, ${betaNumel}`);
    }
    const x = this.materialize();
    const g = gamma.materialize();
    const b = beta.materialize();
    const outId = getDispatcher().nextTensorId();
    getDispatcher().allocate(outId, this.numElements() * 4);
    getDispatcher().runOp("STYLE_AFFINE", [x.id, g.id, b.id], outId, {
      axisSize: C,
      innerSize: inner,
    });
    if (x !== this) x.dispose();
    if (g !== gamma) g.dispose();
    if (b !== beta) b.dispose();
    return new Tensor(outId, this.shape.slice(), false);
  }

  gelu(): Tensor {
    return this.runUnaryOp("GELU");
  }

  sqrt(): Tensor {
    return this.runUnaryOp("SQRT");
  }

  rsqrt(): Tensor {
    return this.runUnaryOp("RSQRT");
  }

  sigmoid(): Tensor {
    return this.runUnaryOp("SIGMOID");
  }

  leaky_relu(negativeSlope = 0.01): Tensor {
    const outId = getDispatcher().nextTensorId();
    const size = this.numElements() * 4;
    getDispatcher().allocate(outId, size);
    getDispatcher().runOp("LEAKY_RELU", [this.id], outId, {
      shape: this.shape,
      strides: this.strides,
      negativeSlope,
    });
    const shouldGrad = GradMode.enabled && this.requiresGrad;
    const out = new Tensor(outId, this.shape, shouldGrad);
    if (shouldGrad) {
      out.op = "LEAKY_RELU";
      out.prev = [this];
      out.params = { negativeSlope };
    }
    return out;
  }

  silu(): Tensor {
    return this.runUnaryOp("SILU");
  }

  // RMSNorm over the last dim: y = x * rsqrt(mean(x²) + eps) * weight.
  // Inference-only (no autograd wiring).
  rms_norm(weight: Tensor, eps: number): Tensor {
    const rank = this.shape.length;
    if (rank < 1) throw new Error(`rms_norm: needs at least 1D input, got shape [${this.shape}]`);
    const n = this.shape[rank - 1];
    const weightNumel = weight.shape.reduce((a, b) => a * b, 1);
    if (weightNumel !== n) {
      throw new Error(`rms_norm: weight numel ${weightNumel} must equal last dim ${n}`);
    }
    const x = this.materialize();
    const w = weight.materialize();
    let m = 1;
    for (let i = 0; i < rank - 1; i++) m *= this.shape[i];

    const outId = getDispatcher().nextTensorId();
    getDispatcher().allocate(outId, x.numElements() * 4);
    getDispatcher().runOp("RMS_NORM", [x.id, w.id], outId, { m, n, eps });

    if (x !== this) x.dispose();
    if (w !== weight) w.dispose();
    return new Tensor(outId, this.shape.slice(), false);
  }

  // Half-split RoPE (HF Llama convention). Input is [..., T, D] with D even;
  // cos/sin caches are [T, D/2] where T matches the seq dim. Position offset
  // is handled by the caller via slicing cos/sin. Inference-only.
  rope(cos: Tensor, sin: Tensor): Tensor {
    const rank = this.shape.length;
    if (rank < 2) throw new Error(`rope: needs at least 2D input, got shape [${this.shape}]`);
    const d = this.shape[rank - 1];
    const tSeq = this.shape[rank - 2];
    if (d % 2 !== 0) throw new Error(`rope: last dim must be even, got ${d}`);
    const dHalf = d / 2;
    if (cos.shape.length !== 2 || cos.shape[0] !== tSeq || cos.shape[1] !== dHalf) {
      throw new Error(`rope: cos shape must be [${tSeq}, ${dHalf}], got [${cos.shape}]`);
    }
    if (sin.shape.length !== 2 || sin.shape[0] !== tSeq || sin.shape[1] !== dHalf) {
      throw new Error(`rope: sin shape must be [${tSeq}, ${dHalf}], got [${sin.shape}]`);
    }

    const x = this.materialize();
    const c = cos.materialize();
    const s = sin.materialize();
    let n = 1;
    for (let i = 0; i < rank - 2; i++) n *= this.shape[i];
    const m = n * tSeq;

    const outId = getDispatcher().nextTensorId();
    getDispatcher().allocate(outId, x.numElements() * 4);
    getDispatcher().runOp("ROPE", [x.id, c.id, s.id], outId, { m, tSeq, dHalf });

    if (x !== this) x.dispose();
    if (c !== cos) c.dispose();
    if (s !== sin) s.dispose();
    return new Tensor(outId, this.shape.slice(), false);
  }

  // Causal-masked softmax over the last dim. Row r sees only cols [0, pastLen+r]
  // (clamped to n-1); disallowed cols are zeroed. pastLen=0 is standard prefill
  // where scores are square [T, T]. Inference-only.
  causal_softmax(pastLen = 0): Tensor {
    const rank = this.shape.length;
    if (rank < 2) throw new Error(`causal_softmax: needs at least 2D input, got shape [${this.shape}]`);
    const n = this.shape[rank - 1];
    const m = this.numElements() / n;

    const x = this.materialize();
    const outId = getDispatcher().nextTensorId();
    getDispatcher().allocate(outId, x.numElements() * 4);
    getDispatcher().runOp("CAUSAL_SOFTMAX", [x.id], outId, { m, n, pastLen });

    if (x !== this) x.dispose();
    return new Tensor(outId, this.shape.slice(), false);
  }

  matmul(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error(`MatMul requires 2D tensors. Got ${this.shape} and ${other.shape}`);
    }
    if (this.shape[1] !== other.shape[0]) {
      throw new Error(`Shape mismatch for MatMul: ${this.shape} vs ${other.shape}`);
    }

    const m = this.shape[0];
    const k = this.shape[1];
    const n = other.shape[1];
    const outShape = [m, n];

    const outId = getDispatcher().nextTensorId();
    const size = m * n * 4;

    getDispatcher().allocate(outId, size);
    getDispatcher().runOp("MATMUL", [this.id, other.id], outId, {
      m,
      n,
      k,
      stridesA: this.strides,
      stridesB: other.strides,
    });

    const shouldGrad = GradMode.enabled && (this.requiresGrad || other.requiresGrad);
    const out = new Tensor(outId, outShape, shouldGrad);
    if (shouldGrad) {
      out.op = "MATMUL";
      out.prev = [this, other];
    }

    return out;
  }

  // Batched matmul: this [B, M, K] × other [B, K, N] → [B, M, N].
  // Kernels assume contiguous row-major operands; strided inputs are materialized.
  bmm(other: Tensor): Tensor {
    if (this.shape.length !== 3 || other.shape.length !== 3) {
      throw new Error(`bmm requires 3D tensors. Got ${this.shape} and ${other.shape}`);
    }
    if (this.shape[0] !== other.shape[0]) {
      throw new Error(`bmm batch mismatch: ${this.shape[0]} vs ${other.shape[0]}`);
    }
    if (this.shape[2] !== other.shape[1]) {
      throw new Error(`bmm inner dim mismatch: ${this.shape} vs ${other.shape}`);
    }

    const a = this.materialize();
    const b = other.materialize();

    const batchCount = a.shape[0];
    const m = a.shape[1];
    const k = a.shape[2];
    const n = b.shape[2];
    const outShape = [batchCount, m, n];

    const outId = getDispatcher().nextTensorId();
    getDispatcher().allocate(outId, batchCount * m * n * 4);
    getDispatcher().runOp("BMM", [a.id, b.id], outId, { batchCount, m, n, k });

    if (a !== this) a.dispose();
    if (b !== other) b.dispose();

    const shouldGrad = GradMode.enabled && (this.requiresGrad || other.requiresGrad);
    const out = new Tensor(outId, outShape, shouldGrad);
    if (shouldGrad) {
      out.op = "BMM";
      out.prev = [this, other];
    }

    return out;
  }

  // 1-D convolution over the last dim of a [B, C_in, L_in] input tensor.
  // Weight is [C_out, C_in/groups, K]; optional bias is [C_out]. Forward-only.
  conv1d(
    weight: Tensor,
    bias: Tensor | null,
    opts: { stride?: number; padding?: number; dilation?: number; groups?: number } = {},
  ): Tensor {
    const stride = opts.stride ?? 1;
    const padding = opts.padding ?? 0;
    const dilation = opts.dilation ?? 1;
    const groups = opts.groups ?? 1;
    if (this.shape.length !== 3) throw new Error(`conv1d input must be [B, C_in, L], got ${this.shape}`);
    if (weight.shape.length !== 3) throw new Error(`conv1d weight must be [C_out, C_in/G, K], got ${weight.shape}`);
    const [B, Cin, Lin] = this.shape;
    const [Cout, wCinPerG, K] = weight.shape;
    if (Cin % groups !== 0) throw new Error(`conv1d Cin=${Cin} not divisible by groups=${groups}`);
    if (Cout % groups !== 0) throw new Error(`conv1d Cout=${Cout} not divisible by groups=${groups}`);
    if (wCinPerG !== Cin / groups) {
      throw new Error(`conv1d weight Cin/G mismatch: weight ${wCinPerG} vs expected ${Cin / groups}`);
    }
    const Lout = Math.floor((Lin + 2 * padding - dilation * (K - 1) - 1) / stride) + 1;
    if (Lout <= 0) throw new Error(`conv1d yields non-positive output length ${Lout}`);
    if (bias && (bias.shape.length !== 1 || bias.shape[0] !== Cout)) {
      throw new Error(`conv1d bias must be [C_out=${Cout}], got ${bias.shape}`);
    }

    const x = this.materialize();
    const w = weight.materialize();
    const b = bias ? bias.materialize() : null;

    const outId = getDispatcher().nextTensorId();
    getDispatcher().allocate(outId, B * Cout * Lout * 4);
    const ids = b ? [x.id, w.id, b.id] : [x.id, w.id];
    getDispatcher().runOp("CONV1D", ids, outId, {
      batchCount: B,
      Cin,
      Lin,
      Cout,
      K,
      Lout,
      stride,
      padding,
      dilation,
      groups,
      hasBias: !!b,
    });
    if (x !== this) x.dispose();
    if (w !== weight) w.dispose();
    if (b && bias && b !== bias) b.dispose();
    return new Tensor(outId, [B, Cout, Lout], false);
  }

  // Transposed 1-D convolution (a.k.a. fractionally-strided conv).
  // Weight is [C_in, C_out/groups, K]. Used for upsampling in HiFiGAN-style vocoders.
  convTranspose1d(
    weight: Tensor,
    bias: Tensor | null,
    opts: {
      stride?: number;
      padding?: number;
      dilation?: number;
      outputPadding?: number;
      groups?: number;
    } = {},
  ): Tensor {
    const stride = opts.stride ?? 1;
    const padding = opts.padding ?? 0;
    const dilation = opts.dilation ?? 1;
    const outputPadding = opts.outputPadding ?? 0;
    const groups = opts.groups ?? 1;
    if (this.shape.length !== 3) throw new Error(`convTranspose1d input must be [B, C_in, L], got ${this.shape}`);
    if (weight.shape.length !== 3) throw new Error(`convTranspose1d weight must be [C_in, C_out/G, K], got ${weight.shape}`);
    const [B, Cin, Lin] = this.shape;
    const [wCin, coutPerG, K] = weight.shape;
    if (Cin !== wCin) throw new Error(`convTranspose1d C_in mismatch: input ${Cin} vs weight ${wCin}`);
    if (Cin % groups !== 0) throw new Error(`convTranspose1d Cin=${Cin} not divisible by groups=${groups}`);
    const Cout = coutPerG * groups;
    const Lout = (Lin - 1) * stride - 2 * padding + dilation * (K - 1) + outputPadding + 1;
    if (Lout <= 0) throw new Error(`convTranspose1d yields non-positive output length ${Lout}`);
    if (bias && (bias.shape.length !== 1 || bias.shape[0] !== Cout)) {
      throw new Error(`convTranspose1d bias must be [C_out=${Cout}], got ${bias.shape}`);
    }

    const x = this.materialize();
    const w = weight.materialize();
    const b = bias ? bias.materialize() : null;

    const outId = getDispatcher().nextTensorId();
    getDispatcher().allocate(outId, B * Cout * Lout * 4);
    const ids = b ? [x.id, w.id, b.id] : [x.id, w.id];
    getDispatcher().runOp("CONV_TRANSPOSE_1D", ids, outId, {
      batchCount: B,
      Cin,
      Lin,
      Cout,
      K,
      Lout,
      stride,
      padding,
      dilation,
      groups,
      hasBias: !!b,
    });
    if (x !== this) x.dispose();
    if (w !== weight) w.dispose();
    if (b && bias && b !== bias) b.dispose();
    return new Tensor(outId, [B, Cout, Lout], false);
  }

  // Fused LSTM step (inference only). `this` = x at the current timestep,
  // shape [B, in_size]. Returns a [B, 2*hidden] tensor packing
  // [h_new || c_new] along the last dim — caller slices into two views.
  //
  // Gate order matches PyTorch nn.LSTM: i, f, g, o. Weights are
  // weight_ih [4*hidden, in_size] and weight_hh [4*hidden, hidden];
  // biases [4*hidden] each.
  lstmStep(
    h: Tensor,
    c: Tensor,
    weightIh: Tensor,
    weightHh: Tensor,
    biasIh: Tensor,
    biasHh: Tensor,
  ): Tensor {
    if (this.shape.length !== 2) throw new Error(`lstmStep x must be [B, in_size], got ${this.shape}`);
    const [B, inSize] = this.shape;
    if (h.shape.length !== 2 || h.shape[0] !== B) throw new Error(`lstmStep h shape ${h.shape} incompatible with x ${this.shape}`);
    if (c.shape.length !== 2 || c.shape[0] !== B || c.shape[1] !== h.shape[1]) {
      throw new Error(`lstmStep c shape ${c.shape} must match h ${h.shape}`);
    }
    const hidden = h.shape[1];
    if (weightIh.shape.length !== 2 || weightIh.shape[0] !== 4 * hidden || weightIh.shape[1] !== inSize) {
      throw new Error(`lstmStep weight_ih shape ${weightIh.shape} expected [${4 * hidden}, ${inSize}]`);
    }
    if (weightHh.shape.length !== 2 || weightHh.shape[0] !== 4 * hidden || weightHh.shape[1] !== hidden) {
      throw new Error(`lstmStep weight_hh shape ${weightHh.shape} expected [${4 * hidden}, ${hidden}]`);
    }
    if (biasIh.shape.length !== 1 || biasIh.shape[0] !== 4 * hidden) {
      throw new Error(`lstmStep bias_ih shape ${biasIh.shape} expected [${4 * hidden}]`);
    }
    if (biasHh.shape.length !== 1 || biasHh.shape[0] !== 4 * hidden) {
      throw new Error(`lstmStep bias_hh shape ${biasHh.shape} expected [${4 * hidden}]`);
    }

    const x = this.materialize();
    const hM = h.materialize();
    const cM = c.materialize();
    const wIh = weightIh.materialize();
    const wHh = weightHh.materialize();
    const bIh = biasIh.materialize();
    const bHh = biasHh.materialize();

    const outId = getDispatcher().nextTensorId();
    getDispatcher().allocate(outId, B * 2 * hidden * 4);
    getDispatcher().runOp(
      "LSTM_STEP",
      [x.id, hM.id, cM.id, wIh.id, wHh.id, bIh.id, bHh.id],
      outId,
      { batchSize: B, hidden, inSize },
    );
    if (x !== this) x.dispose();
    if (hM !== h) hM.dispose();
    if (cM !== c) cM.dispose();
    if (wIh !== weightIh) wIh.dispose();
    if (wHh !== weightHh) wHh.dispose();
    if (bIh !== biasIh) bIh.dispose();
    if (bHh !== biasHh) bHh.dispose();
    return new Tensor(outId, [B, 2 * hidden], false);
  }

  // Direct-write variant: writes h_new and c_new into pre-allocated tensors
  // at the given element offsets. No new tensor allocation. Used by BiLSTM to
  // stack outputs into a [B, T, H] buffer without a per-step concat.
  lstmStepInto(
    h: Tensor,
    c: Tensor,
    weightIh: Tensor,
    weightHh: Tensor,
    biasIh: Tensor,
    biasHh: Tensor,
    hOut: Tensor,
    hOutOffsetElements: number,
    cOut: Tensor,
    cOutOffsetElements: number,
  ): void {
    if (this.shape.length !== 2) throw new Error(`lstmStepInto x must be [B, in_size], got ${this.shape}`);
    const [B, inSize] = this.shape;
    const hidden = h.shape[1];

    const x = this.materialize();
    const hM = h.materialize();
    const cM = c.materialize();
    const wIh = weightIh.materialize();
    const wHh = weightHh.materialize();
    const bIh = biasIh.materialize();
    const bHh = biasHh.materialize();

    // The dispatcher pulls hOut's real heap offset from its registry (Tensor.offset
    // on the JS side is 0 for freshly-allocated tensors). cOut is passed as an 8th
    // "input" so the same lookup gives its real offset too.
    getDispatcher().runOp(
      "LSTM_STEP",
      [x.id, hM.id, cM.id, wIh.id, wHh.id, bIh.id, bHh.id, cOut.id],
      hOut.id,
      {
        batchSize: B,
        hidden,
        inSize,
        hNewOffElements: hOutOffsetElements,
        cNewOffElements: cOutOffsetElements,
      },
    );
    if (x !== this) x.dispose();
    if (hM !== h) hM.dispose();
    if (cM !== c) cM.dispose();
    if (wIh !== weightIh) wIh.dispose();
    if (wHh !== weightHh) wHh.dispose();
    if (bIh !== biasIh) bIh.dispose();
    if (bHh !== biasHh) bHh.dispose();
  }

  embedding(indices: Tensor): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`Embedding weights must be 2D, got ${this.shape}`);
    }

    const embeddingDim = this.shape[1];
    const outShape = [...indices.shape, embeddingDim];

    const outId = getDispatcher().nextTensorId();
    const size = outShape.reduce((a, b) => a * b, 1) * 4;

    getDispatcher().allocate(outId, size);
    getDispatcher().runOp("EMBEDDING", [this.id, indices.id], outId, {
      embeddingDim,
    });

    const shouldGrad = GradMode.enabled && this.requiresGrad;
    const out = new Tensor(outId, outShape, shouldGrad);

    if (shouldGrad) {
      out.op = "EMBEDDING";
      out.prev = [this, indices];
      out.params = { embeddingDim };
    }

    return out;
  }

  // Zero-copy: swap two dims by permuting shape+strides. Defaults to last two
  // (matches PyTorch's `Tensor.T` behavior for rank-2 and `torch.transpose`).
  transpose(dim0 = -2, dim1 = -1): Tensor {
    const rank = this.shape.length;
    if (rank < 2) throw new Error(`Transpose requires rank >= 2, got shape ${this.shape}`);
    const d0 = dim0 < 0 ? rank + dim0 : dim0;
    const d1 = dim1 < 0 ? rank + dim1 : dim1;
    if (d0 < 0 || d0 >= rank || d1 < 0 || d1 >= rank) {
      throw new Error(`Invalid transpose dims (${dim0}, ${dim1}) for rank ${rank}`);
    }
    const outShape = this.shape.slice();
    const outStrides = this.strides.slice();
    [outShape[d0], outShape[d1]] = [outShape[d1], outShape[d0]];
    [outStrides[d0], outStrides[d1]] = [outStrides[d1], outStrides[d0]];

    const viewId = getDispatcher().nextTensorId();
    getDispatcher().allocateView(viewId, this.id);

    const shouldGrad = GradMode.enabled && this.requiresGrad;
    const out = new Tensor(viewId, outShape, shouldGrad, this.offset);
    out.strides = outStrides;

    if (shouldGrad) {
      out.op = "TRANSPOSE";
      out.prev = [this];
      out.params = { axis: d0, axisSize: d1 };
    } else {
      // Share buffer with `this`; keep it alive against GC.
      out.prev = [this];
    }

    return out;
  }

  reshape(newShape: number[]): Tensor {
    // Reshape is a zero-copy view over row-major memory; non-contiguous inputs
    // (e.g. from transpose) can't be reshaped directly, so materialize first.
    const src = this.isContiguous() ? this : this.materialize();
    const total = src.numElements();
    let unknown = -1;
    let known = 1;
    for (let i = 0; i < newShape.length; i++) {
      if (newShape[i] === -1) {
        if (unknown !== -1) throw new Error(`Reshape supports at most one -1: ${newShape}`);
        unknown = i;
      } else {
        known *= newShape[i];
      }
    }
    const resolved = newShape.slice();
    if (unknown !== -1) {
      if (known === 0 || total % known !== 0) {
        throw new Error(`Reshape ${src.shape} to ${newShape}: not divisible by ${known}`);
      }
      resolved[unknown] = total / known;
    } else if (known !== total) {
      throw new Error(`Reshape size mismatch: ${src.shape} (${total}) vs ${newShape} (${known})`);
    }

    const viewId = getDispatcher().nextTensorId();
    getDispatcher().allocateView(viewId, src.id);

    const shouldGrad = GradMode.enabled && src.requiresGrad;
    const out = new Tensor(viewId, resolved, shouldGrad, src.offset);
    if (shouldGrad) {
      out.op = "RESHAPE";
      out.prev = [src];
    } else {
      // View into `src`'s buffer. When src is a materialize result from a
      // non-contig reshape, nothing else holds it; keep it alive against GC.
      out.prev = [src];
    }

    return out;
  }

  toArray(clone = true): Promise<Float32Array> {
    const tensor = this.materialize();
    return clone ? getDispatcher().read(tensor.id) : getDispatcher().readView(tensor.id);
  }

  async item(): Promise<number> {
    const arr = await this.toArray();
    return arr[0];
  }

  withGrad(): Tensor {
    this.enableGrad();
    return this;
  }

  enableGrad() {
    this.requiresGrad = true;
  }

  disableGrad() {
    this.requiresGrad = false;
  }

  backward() {
    if (!this.requiresGrad) return;

    _backwardTracking = new Set();
    try {
      this.backwardImpl();
    } finally {
      this.cleanupBackwardIntermediates();
    }
  }

  private cleanupBackwardIntermediates() {
    const tracked = _backwardTracking;
    _backwardTracking = null;
    if (!tracked) return;

    const keep = new Set<string>();

    // Walk the topo and keep every final .grad tensor
    const visited = new Set<string>();
    const collect = (v: Tensor) => {
      if (visited.has(v.id)) return;
      visited.add(v.id);
      if (v.grad) keep.add(v.grad.id);
      for (const child of v.prev) collect(child);
    };
    collect(this);

    for (const t of tracked) {
      if (!keep.has(t.id)) t.dispose();
    }
  }

  private backwardImpl() {
    const topo: Tensor[] = [];
    const visited = new Set<string>();

    const buildTopo = (v: Tensor) => {
      if (visited.has(v.id)) return;
      visited.add(v.id);
      for (const child of v.prev) {
        buildTopo(child);
      }
      topo.push(v);
    };
    buildTopo(this);

    this.grad = Tensor.create(this.shape, false, "FILL", { value: 1.0 });

    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      if (!v.grad) continue;

      if (v.op === "ADD") {
        const [a, b] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad);
        if (b.requiresGrad) b.addGrad(v.grad);
      } else if (v.op === "SUB") {
        const [a, b] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad);
        if (b.requiresGrad) b.addGrad(v.grad.neg());
      } else if (v.op === "MUL") {
        const [a, b] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad.mul(b));
        if (b.requiresGrad) b.addGrad(v.grad.mul(a));
      } else if (v.op === "DIV") {
        const [a, b] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad.div(b));
        if (b.requiresGrad) b.addGrad(v.grad.mul(a).div(b.mul(b)).neg());
      } else if (v.op === "RELU") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          const outId = getDispatcher().nextTensorId();
          getDispatcher().allocate(outId, a.numElements() * 4);
          getDispatcher().runOp("RELU_BACKWARD", [a.id, v.grad.id], outId, {
            shape: a.shape,
            strides: a.strides,
          });
          const g = new Tensor(outId, a.shape, false);
          a.addGrad(g);
        }
      } else if (v.op === "EXP") {
        const [a] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad.mul(v));
      } else if (v.op === "TANH") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          // gradInput = gradOutput * (1 - output^2); inputs: [output, gradOutput]
          const gradId = getDispatcher().nextTensorId();
          const size = a.numElements() * 4;
          getDispatcher().allocate(gradId, size);
          getDispatcher().runOp("TANH_BACKWARD", [v.id, v.grad.id], gradId);
          const gradTensor = new Tensor(gradId, a.shape, false);
          a.addGrad(gradTensor);
        }
      } else if (v.op === "GELU") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          const gradId = getDispatcher().nextTensorId();
          getDispatcher().allocate(gradId, a.numElements() * 4);
          getDispatcher().runOp("GELU_BACKWARD", [a.id, v.grad.id], gradId, {
            shape: a.shape,
            strides: a.strides,
          });
          a.addGrad(new Tensor(gradId, a.shape, false));
        }
      } else if (v.op === "SQRT") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          const gradId = getDispatcher().nextTensorId();
          getDispatcher().allocate(gradId, a.numElements() * 4);
          getDispatcher().runOp("SQRT_BACKWARD", [v.id, v.grad.id], gradId);
          a.addGrad(new Tensor(gradId, a.shape, false));
        }
      } else if (v.op === "RSQRT") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          const gradId = getDispatcher().nextTensorId();
          getDispatcher().allocate(gradId, a.numElements() * 4);
          getDispatcher().runOp("RSQRT_BACKWARD", [v.id, v.grad.id], gradId);
          a.addGrad(new Tensor(gradId, a.shape, false));
        }
      } else if (v.op === "SIGMOID") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          const gradId = getDispatcher().nextTensorId();
          getDispatcher().allocate(gradId, a.numElements() * 4);
          getDispatcher().runOp("SIGMOID_BACKWARD", [v.id, v.grad.id], gradId);
          a.addGrad(new Tensor(gradId, a.shape, false));
        }
      } else if (v.op === "LEAKY_RELU") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          const gradId = getDispatcher().nextTensorId();
          getDispatcher().allocate(gradId, a.numElements() * 4);
          getDispatcher().runOp("LEAKY_RELU_BACKWARD", [a.id, v.grad.id], gradId, {
            negativeSlope: v.params.negativeSlope ?? 0.01,
          });
          a.addGrad(new Tensor(gradId, a.shape, false));
        }
      } else if (v.op === "SILU") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          const gradId = getDispatcher().nextTensorId();
          getDispatcher().allocate(gradId, a.numElements() * 4);
          getDispatcher().runOp("SILU_BACKWARD", [a.id, v.grad.id], gradId);
          a.addGrad(new Tensor(gradId, a.shape, false));
        }
      } else if (v.op === "LOG") {
        const [a] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad.div(a));
      } else if (v.op === "SOFTMAX") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          const gradId = getDispatcher().nextTensorId();
          const size = a.numElements() * 4;
          getDispatcher().allocate(gradId, size);
          const m = v.shape[0];
          const n = v.shape[1];
          getDispatcher().runOp("SOFTMAX_BACKWARD", [v.id, v.grad.id], gradId, { m, n });
          const gradTensor = new Tensor(gradId, a.shape, false);
          a.addGrad(gradTensor);
        }
      } else if (v.op === "MATMUL") {
        const [a, b] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad.matmul(b.transpose()));
        if (b.requiresGrad) b.addGrad(a.transpose().matmul(v.grad));
      } else if (v.op === "BMM") {
        const [a, b] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad.bmm(b.transpose(-1, -2)));
        if (b.requiresGrad) b.addGrad(a.transpose(-1, -2).bmm(v.grad));
      } else if (v.op === "EMBEDDING") {
        const [weights, indices] = v.prev;
        if (weights.requiresGrad) {
          const gradWeightsId = getDispatcher().nextTensorId();
          const size = weights.numElements() * 4;
          getDispatcher().allocate(gradWeightsId, size);
          getDispatcher().runOp("FILL", [], gradWeightsId, { value: 0 });

          getDispatcher().runOp("EMBEDDING_BACKWARD", [indices.id, v.grad.id], gradWeightsId, {
            embeddingDim: v.params.embeddingDim,
          });

          const gradWeights = new Tensor(gradWeightsId, weights.shape, false);
          weights.addGrad(gradWeights);
        }
      } else if (v.op === "TRANSPOSE") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          const d0 = v.params?.axis ?? -2;
          const d1 = v.params?.axisSize ?? -1;
          a.addGrad(v.grad.transpose(d0, d1));
        }
      } else if (v.op === "RESHAPE") {
        const [a] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad.reshape(a.shape));
      } else if (v.op === "NEG") {
        const [a] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad.neg());
      } else if (v.op === "MATERIALIZE") {
        const [a] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad);
      } else if (v.op === "SUM") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          a.addGrad(v.grad);
        }
      } else if (v.op === "SUM_AXIS") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          let grad = v.grad;
          if (v.shape.length < a.shape.length) {
            const axis = v.params.axis ?? 0;
            const newShape = [...v.shape];
            newShape.splice(axis, 0, 1);
            grad = grad.reshape(newShape);
          }
          const zeros = Tensor.zeros(a.shape);
          const expanded = zeros.add(grad);
          a.addGrad(expanded);
        }
      }
    }
  } // end backwardImpl

  private addGrad(g: Tensor) {
    if (g.numElements() === 1) {
      if (!this.grad) {
        const zeros = Tensor.zeros(this.shape);
        const outId = getDispatcher().nextTensorId();
        const size = this.numElements() * 4;
        getDispatcher().allocate(outId, size);
        getDispatcher().runOp("ADD_SCALAR_TENSOR", [zeros.id, g.id], outId);
        this.grad = new Tensor(outId, this.shape, false);
      } else {
        const outId = getDispatcher().nextTensorId();
        const size = this.numElements() * 4;
        getDispatcher().allocate(outId, size);
        getDispatcher().runOp("ADD_SCALAR_TENSOR", [this.grad.id, g.id], outId);
        this.grad = new Tensor(outId, this.shape, false);
      }
      return;
    }

    const processedG = this.reshapeGrad(g);

    if (this.shapeEquals(processedG.shape)) {
      if (!this.grad) {
        this.grad = processedG;
      } else {
        this.grad = this.grad.add(processedG);
      }
    } else {
      throw new Error(`Gradient shape mismatch: ${this.shape} vs ${processedG.shape}`);
    }
  }

  private reshapeGrad(g: Tensor): Tensor {
    if (this.shapeEquals(g.shape)) return g;

    let out = g;

    // Extra dims (e.g. [2, 3] -> [3])
    while (out.shape.length > this.shape.length) {
      out = out.sum(0, false);
    }

    // Broadcasted dims (e.g. [1, 3] vs [2, 3])
    for (let i = 0; i < this.shape.length; i++) {
      if (this.shape[i] === 1 && out.shape[i] !== 1) {
        out = out.sum(i, true);
      }
    }

    return out;
  }

  neg(): Tensor {
    return this.runUnaryOp("NEG");
  }

  sum(axis?: number, keepDim = false): Tensor {
    if (axis === undefined) {
      const input = this.materialize();

      const outId = getDispatcher().nextTensorId();
      const size = 4;

      getDispatcher().allocate(outId, size);
      getDispatcher().runOp("SUM", [input.id], outId);
      if (input !== this) input.dispose();

      const shouldGrad = GradMode.enabled && this.requiresGrad;
      const out = new Tensor(outId, [1], shouldGrad);
      if (shouldGrad) {
        out.op = "SUM";
        out.prev = [this];
      }

      return out;
    }

    if (axis < 0) axis += this.shape.length;
    if (axis < 0 || axis >= this.shape.length) {
      throw new Error(`Invalid axis ${axis} for shape ${this.shape}`);
    }

    const input = this.materialize();

    const outShape = input.shape.filter((_, i) => i !== axis);
    const finalShape = keepDim ? input.shape.map((s, i) => (i === axis ? 1 : s)) : outShape;

    const outId = getDispatcher().nextTensorId();
    const size = outShape.reduce((a, b) => a * b, 1) * 4;

    getDispatcher().allocate(outId, size);
    getDispatcher().runOp("SUM_AXIS", [input.id], outId, {
      shape: input.shape,
      strides: input.strides,
      axis,
    });
    if (input !== this) input.dispose();

    const shouldGrad = GradMode.enabled && this.requiresGrad;
    const out = new Tensor(outId, finalShape, shouldGrad);
    if (shouldGrad) {
      out.op = "SUM_AXIS";
      out.prev = [this];
      out.params = { axis, keepDim };
    }

    return out;
  }

  mean(axis?: number, keepDim = false): Tensor {
    if (axis === undefined) {
      const s = this.sum();
      const n = this.numElements();
      return s.div(Tensor.create([1], false, "FILL", { value: n }));
    }
    const axisResolved = axis < 0 ? this.shape.length + axis : axis;
    const N = this.shape[axisResolved];
    const s = this.sum(axis, keepDim);
    return s.div(Tensor.create([1], false, "FILL", { value: N }));
  }

  softmax(axis = -1): Tensor {
    const axisResolved = axis < 0 ? this.shape.length + axis : axis;
    if (this.shape.length === 2 && axisResolved === 1) {
      const input = this.materialize();
      const m = input.shape[0];
      const n = input.shape[1];
      const outId = getDispatcher().nextTensorId();
      const size = input.numElements() * 4;
      getDispatcher().allocate(outId, size);
      getDispatcher().runOp("SOFTMAX", [input.id], outId, { m, n });
      const outShape = input.shape;
      if (input !== this) input.dispose();

      const shouldGrad = GradMode.enabled && this.requiresGrad;
      const out = new Tensor(outId, outShape, shouldGrad);
      if (shouldGrad) {
        out.op = "SOFTMAX";
        out.prev = [this];
        out.params = { m, n };
      }
      return out;
    }

    const exp = this.exp();
    const sumExp = exp.sum(axis, true);
    return exp.div(sumExp);
  }

  add_(other: Tensor): Tensor {
    this.runBinaryOpInPlace("ADD", other);
    return this;
  }

  sub_(other: Tensor): Tensor {
    this.runBinaryOpInPlace("SUB", other);
    return this;
  }

  mul_(other: Tensor): Tensor {
    this.runBinaryOpInPlace("MUL", other);
    return this;
  }

  div_(other: Tensor): Tensor {
    this.runBinaryOpInPlace("DIV", other);
    return this;
  }

  zero_(): Tensor {
    getDispatcher().runOp("FILL", [], this.id, { value: 0 });
    return this;
  }

  private runBinaryOpInPlace(op: string, other: Tensor) {
    const outShape = Tensor.broadcastShapes(this.shape, other.shape);
    if (!this.shapeEquals(outShape)) {
      throw new Error(
        `In-place op requires output shape to match. Got ${this.shape} vs broadcasted ${outShape}`,
      );
    }

    const stridesA = Tensor.getBroadcastStrides(this.shape, this.strides, outShape);
    const stridesB = Tensor.getBroadcastStrides(other.shape, other.strides, outShape);

    getDispatcher().runOp(op, [this.id, other.id], this.id, {
      shape: outShape,
      stridesA,
      stridesB,
    });
  }

  private runBinaryOp(op: string, other: Tensor): Tensor {
    const outShape = Tensor.broadcastShapes(this.shape, other.shape);
    const stridesA = Tensor.getBroadcastStrides(this.shape, this.strides, outShape);
    const stridesB = Tensor.getBroadcastStrides(other.shape, other.strides, outShape);

    const outId = getDispatcher().nextTensorId();
    const size = outShape.reduce((a, b) => a * b, 1) * 4;

    getDispatcher().allocate(outId, size);
    getDispatcher().runOp(op, [this.id, other.id], outId, {
      shape: outShape,
      stridesA,
      stridesB,
    });

    const shouldGrad = GradMode.enabled && (this.requiresGrad || other.requiresGrad);
    const out = new Tensor(outId, outShape, shouldGrad);
    if (shouldGrad) {
      out.op = op;
      out.prev = [this, other];
    }

    return out;
  }

  private runUnaryOp(op: string): Tensor {
    const outId = getDispatcher().nextTensorId();
    const size = this.numElements() * 4;

    getDispatcher().allocate(outId, size);
    getDispatcher().runOp(op, [this.id], outId, {
      shape: this.shape,
      strides: this.strides,
    });

    const shouldGrad = GradMode.enabled && this.requiresGrad;
    const out = new Tensor(outId, this.shape, shouldGrad);
    if (shouldGrad) {
      out.op = op;
      out.prev = [this];
    }

    return out;
  }

  private shapeEquals(other: number[]): boolean {
    if (this.shape.length !== other.length) return false;
    for (let i = 0; i < this.shape.length; i++) {
      if (this.shape[i] !== other[i]) return false;
    }
    return true;
  }

  private numElements(): number {
    return this.shape.reduce((a, b) => a * b, 1);
  }

  private static broadcastShapes(shapeA: number[], shapeB: number[]): number[] {
    const ndimA = shapeA.length;
    const ndimB = shapeB.length;
    const ndim = Math.max(ndimA, ndimB);
    const outShape = new Array(ndim);

    for (let i = 0; i < ndim; i++) {
      const dimA = i < ndim - ndimA ? 1 : shapeA[i - (ndim - ndimA)];
      const dimB = i < ndim - ndimB ? 1 : shapeB[i - (ndim - ndimB)];

      if (dimA !== dimB && dimA !== 1 && dimB !== 1) {
        throw new Error(`Shapes ${shapeA} and ${shapeB} are not broadcastable`);
      }
      outShape[i] = Math.max(dimA, dimB);
    }
    return outShape;
  }

  private static getBroadcastStrides(
    shape: number[],
    strides: number[],
    outShape: number[],
  ): number[] {
    const ndim = outShape.length;
    const ndimIn = shape.length;
    const outStrides = new Array(ndim).fill(0);

    for (let i = 0; i < ndim; i++) {
      const dimIn = i - (ndim - ndimIn);

      if (dimIn >= 0) {
        if (shape[dimIn] === 1) {
          outStrides[i] = 0;
        } else {
          outStrides[i] = strides[dimIn];
        }
      } else {
        outStrides[i] = 0;
      }
    }
    return outStrides;
  }
}
