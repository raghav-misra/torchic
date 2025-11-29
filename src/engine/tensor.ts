import { dispatcherInstance } from "./init";
type NestedArray = number | NestedArray[];

// Automatic memory management
const registry = new FinalizationRegistry((id: string) => {
  dispatcherInstance!.free(id);
});

// Global state for gradient computation
export const GradMode = {
  enabled: true,
};

// Global state for manual memory management tracking
let _activeTracking: Set<Tensor> | null = null;

export async function noGrad<T>(fn: () => Promise<T>): Promise<T> {
  const prev = GradMode.enabled;
  GradMode.enabled = false;
  return fn().finally(() => {
    GradMode.enabled = prev;
  });
}

// ...existing code...

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

export interface OpParams {
  axis?: number;
  keepDim?: boolean;
  m?: number;
  n?: number;
  k?: number;
  value?: number;
  shape?: number[];
  strides?: number[];
  stridesA?: number[];
  stridesB?: number[];
  embeddingDim?: number;
}

export class Tensor {
  id: string;
  shape: number[];
  strides: number[];
  offset: number; // Byte offset in SharedArrayBuffer
  requiresGrad: boolean;
  grad: Tensor | null = null;

  // Graph for Autograd
  op: string | null = null;
  prev: Tensor[] = [];
  params: OpParams = {};
  isDisposed: boolean = false;

  constructor(
    id: string,
    shape: number[],
    requiresGrad: boolean = false,
    offset: number = 0,
    strides?: number[]
  ) {
    this.id = id;
    this.shape = shape;
    this.strides = strides ? strides.slice() : Tensor.computeStrides(shape);
    this.offset = offset;
    this.requiresGrad = requiresGrad;

    // Register for automatic cleanup when this JS object is GC'd
    registry.register(this, this.id, this);

    if (_activeTracking) {
      _activeTracking.add(this);
    }
  }
  /**
   * Returns a zero-copy n-dimensional slice view of this tensor.
   * @param ranges Array of [start, end) for each dimension
   */
  slice(ranges: Array<[number, number]>): Tensor {
    if (ranges.length !== this.shape.length) {
      throw new Error(
        `slice: ranges length ${ranges.length} does not match tensor rank ${this.shape.length}`
      );
    }
    const newShape = ranges.map(([start, end], i) => {
      if (start < 0 || end > this.shape[i] || start >= end) {
        throw new Error(
          `Invalid slice range [${start}, ${end}) for dimension ${i} with size ${this.shape[i]}`
        );
      }
      return end - start;
    });
    // Compute relative start in elements (sum over dims start*stride)
    const relativeStartElements = ranges.reduce(
      (acc, [start], i) => acc + start * this.strides[i],
      0
    );

    // Request coordinator to create a view with the relative byte offset
    const viewId = dispatcherInstance!.nextTensorId();
    const relativeOffsetBytes = relativeStartElements * 4; // elements -> bytes
    dispatcherInstance!.allocateView(viewId, this.id, relativeOffsetBytes);

    // Create Tensor with the correct (JS-side) offset for convenience
    const newOffset = this.offset + relativeOffsetBytes;
    return new Tensor(
      viewId,
      newShape,
      this.requiresGrad,
      newOffset,
      this.strides
    );
  }

  /**
   * Sets the value at the given n-dimensional indices.
   * @param indices Array of indices for each dimension
   * @param value Value to set
   */
  set(indices: number[], value: number) {
    if (indices.length !== this.shape.length) {
      throw new Error(
        `set: indices length ${indices.length} does not match tensor rank ${this.shape.length}`
      );
    }
    let flatIndex = 0;
    for (let i = 0; i < indices.length; i++) {
      if (indices[i] < 0 || indices[i] >= this.shape[i]) {
        throw new Error(
          `set: index ${indices[i]} out of bounds for dimension ${i} with size ${this.shape[i]}`
        );
      }
      flatIndex += indices[i] * this.strides[i];
    }
    // Adjust for offset
    flatIndex += this.offset / 4; // offset is in bytes, strides are in elements
    dispatcherInstance!.set(this.id, flatIndex, value);
  }

  dispose() {
    if (this.isDisposed) return;
    this.isDisposed = true;
    registry.unregister(this);
    dispatcherInstance!.free(this.id);
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
    // Check if strides match C-contiguous layout
    let expectedStride = 1;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      if (this.strides[i] !== expectedStride) return false;
      expectedStride *= this.shape[i];
    }
    return true;
  }

  private materialize(): Tensor {
    // If already contiguous, return self
    if (this.isContiguous()) return this;

    // Create a new contiguous copy
    const outId = dispatcherInstance!.nextTensorId();
    const size = this.numElements() * 4;

    dispatcherInstance!.allocate(outId, size);
    dispatcherInstance!.runOp("MATERIALIZE", [this.id], outId, {
      shape: this.shape,
      strides: this.strides,
    });

    // Preserve gradient tracking from parent
    const out = new Tensor(outId, this.shape, this.requiresGrad);
    if (this.requiresGrad && GradMode.enabled) {
      out.op = "MATERIALIZE";
      out.prev = [this];
    }
    return out;
  }

  static ones(shape: number[], requiresGrad: boolean = false): Tensor {
    return Tensor.create(shape, requiresGrad, "FILL", { value: 1 });
  }

  static zeros(shape: number[], requiresGrad: boolean = false): Tensor {
    return Tensor.create(shape, requiresGrad, "FILL", { value: 0 });
  }

  static randn(shape: number[], requiresGrad: boolean = false): Tensor {
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
    requiresGrad: boolean = false
  ): Tensor {
    // Robust shape inference and efficient flattening.
    function inferShape(arr: any): number[] {
      if (arr instanceof Float32Array) return [arr.length];
      if (!Array.isArray(arr)) return [];
      const shape: number[] = [];
      let curr = arr;
      while (Array.isArray(curr)) {
        shape.push(curr.length);
        curr = curr[0];
      }
      return shape;
    }

    function flattenInto(arr: any, out: number[]) {
      if (arr instanceof Float32Array) {
        for (let i = 0; i < arr.length; i++) out.push(arr[i]);
        return;
      }
      if (!Array.isArray(arr)) {
        out.push(arr as number);
        return;
      }
      for (const el of arr) flattenInto(el, out);
    }

    let inferredShape = shape;
    let typedData: Float32Array;

    if (inferredShape === undefined) {
      inferredShape = inferShape(data);
      if (data instanceof Float32Array) {
        // Use external Float32Array directly; worker will copy into SharedArrayBuffer
        typedData = data;
      } else {
        const flat: number[] = [];
        flattenInto(data, flat);
        typedData = new Float32Array(flat);
      }
    } else {
      if (data instanceof Float32Array) {
        // Use external Float32Array directly; worker will copy into SharedArrayBuffer
        typedData = data;
      } else {
        const flat: number[] = Array.isArray(data) ? [] : [];
        if (Array.isArray(data)) flattenInto(data, flat);
        else flat.push(data as number);
        typedData = new Float32Array(flat);
      }
    }

    const size = inferredShape.reduce((a, b) => a * b, 1) * 4;
    const id = dispatcherInstance!.nextTensorId();

    dispatcherInstance!.allocate(id, size);
    dispatcherInstance!.write(id, typedData);

    return new Tensor(id, inferredShape!, requiresGrad);
  }

  /**
   * Allocate an uninitialized tensor buffer of the given shape.
   * Useful for reusing the same backing memory across iterations.
   */
  static empty(shape: number[], requiresGrad: boolean = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1) * 4;
    const id = dispatcherInstance!.nextTensorId();
    dispatcherInstance!.allocate(id, size);
    return new Tensor(id, shape, requiresGrad);
  }

  /**
   * Write data into this tensor's backing buffer. `data` should be a
   * Float32Array (or array convertible to it); if shorter than the
   * tensor's storage it will overwrite the prefix.
   */
  write(data: Float32Array | number[]) {
    const arr = data instanceof Float32Array ? data : new Float32Array(data);
    dispatcherInstance!.write(this.id, arr);
  }

  private static create(
    shape: number[],
    requiresGrad: boolean,
    op: string,
    params: OpParams = {}
  ): Tensor {
    const size = shape.reduce((a, b) => a * b, 1) * 4; // 4 bytes per float
    const id = dispatcherInstance!.nextTensorId();

    dispatcherInstance!.allocate(id, size);
    dispatcherInstance!.runOp(op, [], id, params);

    return new Tensor(id, shape, requiresGrad);
  }

  // --- Operations ---

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

  // ...existing code...

  matmul(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error(
        `MatMul requires 2D tensors. Got ${this.shape} and ${other.shape}`
      );
    }
    if (this.shape[1] !== other.shape[0]) {
      throw new Error(
        `Shape mismatch for MatMul: ${this.shape} vs ${other.shape}`
      );
    }

    // Use strided-aware matmul kernel to avoid materializing views where possible.
    const a = this;
    const b = other;

    const m = a.shape[0];
    const k = a.shape[1];
    const n = b.shape[1];
    const outShape = [m, n];

    const outId = dispatcherInstance!.nextTensorId();
    const size = m * n * 4;

    dispatcherInstance!.allocate(outId, size);
    dispatcherInstance!.runOp("MATMUL", [a.id, b.id], outId, {
      m,
      n,
      k,
      stridesA: a.strides,
      stridesB: b.strides,
    });

    const shouldGrad =
      GradMode.enabled && (this.requiresGrad || other.requiresGrad);
    const out = new Tensor(outId, outShape, shouldGrad);
    if (shouldGrad) {
      out.op = "MATMUL";
      out.prev = [this, other]; // Keep original tensors in graph
    }

    return out;
  }

  embedding(indices: Tensor): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`Embedding weights must be 2D, got ${this.shape}`);
    }

    const embeddingDim = this.shape[1];
    const outShape = [...indices.shape, embeddingDim];

    const outId = dispatcherInstance!.nextTensorId();
    const size = outShape.reduce((a, b) => a * b, 1) * 4;

    dispatcherInstance!.allocate(outId, size);
    dispatcherInstance!.runOp("EMBEDDING", [this.id, indices.id], outId, {
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

  transpose(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error("Transpose only supported for 2D tensors");
    }
    const [m, n] = this.shape;
    const outShape = [n, m];

    // Zero-copy: create a view with new ID but swap strides
    const viewId = dispatcherInstance!.nextTensorId();
    dispatcherInstance!.allocateView(viewId, this.id);

    const shouldGrad = GradMode.enabled && this.requiresGrad;
    const out = new Tensor(viewId, outShape, shouldGrad, this.offset);
    // Swap strides: if original was [stride0, stride1], transposed is [stride1, stride0]
    out.strides = [this.strides[1], this.strides[0]];

    if (shouldGrad) {
      out.op = "TRANSPOSE";
      out.prev = [this];
    }

    return out;
  }

  reshape(newShape: number[]): Tensor {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== this.numElements()) {
      throw new Error(
        `Reshape size mismatch: ${
          this.shape
        } (${this.numElements()}) vs ${newShape} (${newSize})`
      );
    }

    // Zero-copy: create a view with new ID but same memory offset
    const viewId = dispatcherInstance!.nextTensorId();
    dispatcherInstance!.allocateView(viewId, this.id);

    const shouldGrad = GradMode.enabled && this.requiresGrad;
    const out = new Tensor(viewId, newShape, shouldGrad, this.offset);
    if (shouldGrad) {
      out.op = "RESHAPE";
      out.prev = [this];
    }

    return out;
  }

  // --- Data Access ---

  toArray(clone: boolean = true): Promise<Float32Array> {
    // If non-contiguous (e.g., transposed view), materialize first
    const tensor = this.materialize();
    return clone
      ? dispatcherInstance!.read(tensor.id)
      : dispatcherInstance!.readView(tensor.id);
  }

  async item(): Promise<number> {
    const arr = await this.toArray();
    return arr[0];
  }

  // --- Autograd ---

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

    // 1. Topological Sort
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

    // 2. Initialize Grad
    this.grad = Tensor.create(this.shape, false, "FILL", { value: 1.0 });

    // 3. Reverse Pass
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
          // RELU_BACKWARD: gradInput = (input > 0) ? gradOutput : 0
          const outId = dispatcherInstance!.nextTensorId();
          dispatcherInstance!.allocate(outId, a.numElements() * 4);
          dispatcherInstance!.runOp("RELU_BACKWARD", [a.id, v.grad.id], outId, {
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
          // Use a dedicated TANH_BACKWARD kernel to compute gradInput = gradOutput * (1 - output^2)
          const gradId = dispatcherInstance!.nextTensorId();
          const size = a.numElements() * 4;
          dispatcherInstance!.allocate(gradId, size);
          // inputs: [output (tanh(a)), gradOutput]
          dispatcherInstance!.runOp("TANH_BACKWARD", [v.id, v.grad.id], gradId);
          const gradTensor = new Tensor(gradId, a.shape, false);
          a.addGrad(gradTensor);
        }
      } else if (v.op === "LOG") {
        const [a] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad.div(a));
      } else if (v.op === "SOFTMAX") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          // Use dedicated SOFTMAX_BACKWARD kernel which expects [output, gradOutput]
          const gradId = dispatcherInstance!.nextTensorId();
          const size = a.numElements() * 4;
          dispatcherInstance!.allocate(gradId, size);
          const m = v.shape[0];
          const n = v.shape[1];
          dispatcherInstance!.runOp(
            "SOFTMAX_BACKWARD",
            [v.id, v.grad.id],
            gradId,
            { m, n }
          );
          const gradTensor = new Tensor(gradId, a.shape, false);
          a.addGrad(gradTensor);
        }
      } else if (v.op === "MATMUL") {
        const [a, b] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad.matmul(b.transpose()));
        if (b.requiresGrad) b.addGrad(a.transpose().matmul(v.grad));
      } else if (v.op === "EMBEDDING") {
        const [weights, indices] = v.prev;
        if (weights.requiresGrad) {
          const gradWeightsId = dispatcherInstance!.nextTensorId();
          const size = weights.numElements() * 4;
          dispatcherInstance!.allocate(gradWeightsId, size);
          dispatcherInstance!.runOp("FILL", [], gradWeightsId, { value: 0 });

          dispatcherInstance!.runOp(
            "EMBEDDING_BACKWARD",
            [indices.id, v.grad.id],
            gradWeightsId,
            {
              embeddingDim: v.params.embeddingDim,
            }
          );

          const gradWeights = new Tensor(gradWeightsId, weights.shape, false);
          weights.addGrad(gradWeights);
        }
      } else if (v.op === "TRANSPOSE") {
        const [a] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad.transpose());
      } else if (v.op === "RESHAPE") {
        const [a] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad.reshape(a.shape));
      } else if (v.op === "MATERIALIZE") {
        const [a] = v.prev;
        if (a.requiresGrad) a.addGrad(v.grad);
      } else if (v.op === "SUM") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          // Gradient of sum is 1s expanded to input shape
          // But since we accumulate gradients, we just add the scalar grad to all elements
          // We handle this in addGrad via broadcasting
          a.addGrad(v.grad);
        }
      } else if (v.op === "SUM_AXIS") {
        const [a] = v.prev;
        if (a.requiresGrad) {
          let grad = v.grad;
          // If keepDim was false, we need to restore the dimension
          if (v.shape.length < a.shape.length) {
            const axis = v.params.axis!;
            const newShape = [...v.shape];
            newShape.splice(axis, 0, 1);
            grad = grad.reshape(newShape);
          }
          // Broadcast to a.shape by adding to zeros
          const zeros = Tensor.zeros(a.shape);
          const expanded = zeros.add(grad);
          a.addGrad(expanded);
        }
      }
    }
  }

  private addGrad(g: Tensor) {
    // Optimization: If g is scalar, we don't need to reshape it,
    // and we can handle it via broadcasting immediately.
    if (g.numElements() === 1) {
      if (!this.grad) {
        const zeros = Tensor.zeros(this.shape);
        const outId = dispatcherInstance!.nextTensorId();
        const size = this.numElements() * 4;
        dispatcherInstance!.allocate(outId, size);
        dispatcherInstance!.runOp("ADD_SCALAR_TENSOR", [zeros.id, g.id], outId);
        this.grad = new Tensor(outId, this.shape, false);
      } else {
        const outId = dispatcherInstance!.nextTensorId();
        const size = this.numElements() * 4;
        dispatcherInstance!.allocate(outId, size);
        dispatcherInstance!.runOp(
          "ADD_SCALAR_TENSOR",
          [this.grad.id, g.id],
          outId
        );
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
      throw new Error(
        `Gradient shape mismatch: ${this.shape} vs ${processedG.shape}`
      );
    }
  }

  private reshapeGrad(g: Tensor): Tensor {
    if (this.shapeEquals(g.shape)) return g;

    let out = g;

    // 1. Handle extra dimensions (e.g. [2, 3] -> [3])
    while (out.shape.length > this.shape.length) {
      out = out.sum(0, false);
    }

    // 2. Handle broadcasted dimensions (e.g. [1, 3] vs [2, 3])
    for (let i = 0; i < this.shape.length; i++) {
      if (this.shape[i] === 1 && out.shape[i] !== 1) {
        out = out.sum(i, true);
      }
    }

    return out;
  }

  neg(): Tensor {
    return this.mul(Tensor.create(this.shape, false, "FILL", { value: -1 }));
  }

  sum(axis?: number, keepDim: boolean = false): Tensor {
    if (axis === undefined) {
      // Materialize if non-contiguous
      const input = this.materialize();

      const outId = dispatcherInstance!.nextTensorId();
      const size = 4; // Scalar

      dispatcherInstance!.allocate(outId, size);
      dispatcherInstance!.runOp("SUM", [input.id], outId);

      const shouldGrad = GradMode.enabled && this.requiresGrad;
      const out = new Tensor(outId, [1], shouldGrad);
      if (shouldGrad) {
        out.op = "SUM";
        out.prev = [this]; // Keep original in graph
      }

      return out;
    }

    if (axis < 0) axis += this.shape.length;
    if (axis < 0 || axis >= this.shape.length) {
      throw new Error(`Invalid axis ${axis} for shape ${this.shape}`);
    }

    // Materialize if non-contiguous
    const input = this.materialize();

    const outShape = input.shape.filter((_, i) => i !== axis);
    // If keepDim is true, we want [d0, 1, d2] but the underlying data is flat [d0*d2]
    // The Tensor shape property handles the view.
    const finalShape = keepDim
      ? input.shape.map((s, i) => (i === axis ? 1 : s))
      : outShape;

    const outId = dispatcherInstance!.nextTensorId();
    const size = outShape.reduce((a, b) => a * b, 1) * 4;

    dispatcherInstance!.allocate(outId, size);
    dispatcherInstance!.runOp("SUM_AXIS", [input.id], outId, {
      shape: input.shape,
      strides: input.strides,
      axis,
    });

    const shouldGrad = GradMode.enabled && this.requiresGrad;
    const out = new Tensor(outId, finalShape, shouldGrad);
    if (shouldGrad) {
      out.op = "SUM_AXIS";
      out.prev = [this]; // Keep original in graph
      out.params = { axis, keepDim };
    }

    return out;
  }

  mean(): Tensor {
    const s = this.sum();
    const n = this.numElements();
    return s.div(Tensor.create([1], false, "FILL", { value: n }));
  }

  softmax(axis: number = -1): Tensor {
    // Fast path: 2D softmax over last axis (rows independent)
    const axisResolved = axis < 0 ? this.shape.length + axis : axis;
    if (this.shape.length === 2 && axisResolved === 1) {
      const input = this.materialize();
      const m = input.shape[0];
      const n = input.shape[1];
      const outId = dispatcherInstance!.nextTensorId();
      const size = input.numElements() * 4;
      dispatcherInstance!.allocate(outId, size);
      dispatcherInstance!.runOp("SOFTMAX", [input.id], outId, { m, n });

      const shouldGrad = GradMode.enabled && this.requiresGrad;
      const out = new Tensor(outId, input.shape, shouldGrad);
      if (shouldGrad) {
        out.op = "SOFTMAX";
        out.prev = [this];
        out.params = { m, n };
      }
      return out;
    }

    // Fallback: compose from primitive ops (less efficient)
    const exp = this.exp();
    const sumExp = exp.sum(axis, true);
    return exp.div(sumExp);
  }

  // --- In-Place Operations ---

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
    dispatcherInstance!.runOp("FILL", [], this.id, { value: 0 });
    return this;
  }

  private runBinaryOpInPlace(op: string, other: Tensor) {
    const outShape = Tensor.broadcastShapes(this.shape, other.shape);

    // Ensure output shape matches this tensor's shape (no resizing allowed)
    if (!this.shapeEquals(outShape)) {
      throw new Error(
        `In-place op requires output shape to match. Got ${this.shape} vs broadcasted ${outShape}`
      );
    }

    const stridesA = Tensor.getBroadcastStrides(
      this.shape,
      this.strides,
      outShape
    );
    const stridesB = Tensor.getBroadcastStrides(
      other.shape,
      other.strides,
      outShape
    );

    dispatcherInstance!.runOp(op, [this.id, other.id], this.id, {
      shape: outShape,
      stridesA,
      stridesB,
    });
  }

  // --- Internal Helpers ---

  private runBinaryOp(op: string, other: Tensor): Tensor {
    const outShape = Tensor.broadcastShapes(this.shape, other.shape);
    const stridesA = Tensor.getBroadcastStrides(
      this.shape,
      this.strides,
      outShape
    );
    const stridesB = Tensor.getBroadcastStrides(
      other.shape,
      other.strides,
      outShape
    );

    const outId = dispatcherInstance!.nextTensorId();
    const size = outShape.reduce((a, b) => a * b, 1) * 4;

    dispatcherInstance!.allocate(outId, size);
    dispatcherInstance!.runOp(op, [this.id, other.id], outId, {
      shape: outShape,
      stridesA,
      stridesB,
    });

    const shouldGrad =
      GradMode.enabled && (this.requiresGrad || other.requiresGrad);
    const out = new Tensor(outId, outShape, shouldGrad);
    if (shouldGrad) {
      out.op = op;
      out.prev = [this, other];
    }

    return out;
  }

  private runUnaryOp(op: string): Tensor {
    // Avoid materialize where possible: pass shape/strides into the kernel so
    // it can operate on non-contiguous views directly. Kernels should handle
    // optional shape/strides params.
    const input = this;

    const outId = dispatcherInstance!.nextTensorId();
    const size = input.numElements() * 4;

    dispatcherInstance!.allocate(outId, size);
    dispatcherInstance!.runOp(op, [input.id], outId, {
      shape: input.shape,
      strides: input.strides,
    });

    const shouldGrad = GradMode.enabled && this.requiresGrad;
    const out = new Tensor(outId, input.shape, shouldGrad);
    if (shouldGrad) {
      out.op = op;
      out.prev = [this]; // Keep original in graph
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
    outShape: number[]
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
