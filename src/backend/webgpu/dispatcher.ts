import type { Dispatcher, MemoryStats } from "../dispatcher";
import { MemoryAllocator } from "../memory";
import type { OpParams, TensorId } from "../../shared/types";
import { requestContext } from "./device";
import { buildPipelines, type Pipelines } from "./pipelines";

interface TensorMetadata {
  offset: number;
  size: number;
  isView: boolean;
}

const MATMUL_TILE = 16;
const ELEMENTWISE_TILE = 256;
const ROW_TILE = 64;
const UNIFORM_BUFFER_SIZE = 256;
const SUM_PARTIALS = 64;

// Sentinel free() for maybeMaterialize when the input was already contiguous.
const NOOP_FREE = (): void => undefined;

export class WebGPUDispatcher implements Dispatcher {
  private device: GPUDevice | null = null;
  private queue: GPUQueue | null = null;
  private heap: GPUBuffer | null = null;
  private uniforms: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private pipelines: Pipelines | null = null;
  private allocator: MemoryAllocator | null = null;
  private readonly tensorRegistry = new Map<TensorId, TensorMetadata>();
  private tensorIdCounter = 0;
  private readonly instanceTag = crypto.randomUUID().slice(0, 8);
  private inflight: Promise<void> = Promise.resolve();
  private readonly opCounts = new Map<string, number>();

  async init(_threadCount = 0, memorySizeMB = 256): Promise<void> {
    if (this.device) return;

    const { device, queue } = await requestContext();
    this.device = device;
    this.queue = queue;

    const heapBytes = memorySizeMB * 1024 * 1024;
    const maxBuf = device.limits.maxBufferSize;
    const maxStorage = device.limits.maxStorageBufferBindingSize;
    if (heapBytes > maxBuf || heapBytes > maxStorage) {
      throw new Error(
        `Requested heap ${memorySizeMB} MB exceeds device limits (maxBufferSize=${(maxBuf / 1024 / 1024) | 0} MB, maxStorageBufferBindingSize=${(maxStorage / 1024 / 1024) | 0} MB)`,
      );
    }
    this.heap = device.createBuffer({
      label: "torchic heap",
      size: heapBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    this.uniforms = device.createBuffer({
      label: "torchic uniforms",
      size: UNIFORM_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.pipelines = await buildPipelines(device);
    this.bindGroup = device.createBindGroup({
      layout: this.pipelines.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.heap } },
        { binding: 1, resource: { buffer: this.uniforms } },
      ],
    });

    this.allocator = new MemoryAllocator(heapBytes);
  }

  shutdown(): void {
    this.heap?.destroy();
    this.uniforms?.destroy();
    this.device?.destroy();
    this.heap = null;
    this.uniforms = null;
    this.bindGroup = null;
    this.pipelines = null;
    this.device = null;
    this.queue = null;
    this.allocator = null;
    this.tensorRegistry.clear();
    this.tensorIdCounter = 0;
    this.inflight = Promise.resolve();
  }

  nextTensorId(): TensorId {
    return `${this.instanceTag}_t_${this.tensorIdCounter++}`;
  }

  memoryStats(): MemoryStats {
    if (!this.allocator) return { total: 0, used: 0, free: 0, largestFree: 0, fragments: 0 };
    return this.allocator.getStats();
  }

  opCountSnapshot(): Record<string, number> {
    return Object.fromEntries(this.opCounts);
  }

  resetOpCounts(): void {
    this.opCounts.clear();
  }

  async sync(): Promise<void> {
    await this.inflight;
  }

  allocate(tensorId: TensorId, size: number): void {
    const alloc = this.allocator;
    if (!alloc) throw new Error("Dispatcher not initialized");
    try {
      const offset = alloc.allocate(size);
      this.tensorRegistry.set(tensorId, { offset, size, isView: false });
    } catch (e: unknown) {
      console.error("WebGPU allocation failed:", e instanceof Error ? e.message : e);
    }
  }

  allocateView(tensorId: TensorId, parentId: TensorId, offsetBytes?: number): void {
    const parent = this.tensorRegistry.get(parentId);
    if (!parent) {
      console.error(`Cannot create view: parent tensor ${parentId} not found`);
      return;
    }
    const offset = typeof offsetBytes === "number" ? parent.offset + offsetBytes : parent.offset;
    this.tensorRegistry.set(tensorId, { offset, size: parent.size, isView: true });
  }

  free(tensorId: TensorId): void {
    const meta = this.tensorRegistry.get(tensorId);
    if (!meta) return;
    if (!meta.isView) this.allocator?.free(meta.offset, meta.size);
    this.tensorRegistry.delete(tensorId);
  }

  runOp(op: string, inputs: TensorId[], output: TensorId, params: OpParams = {}): void {
    if (!this.device || !this.queue || !this.bindGroup || !this.uniforms) {
      throw new Error("Dispatcher not initialized");
    }
    this.opCounts.set(op, (this.opCounts.get(op) ?? 0) + 1);

    const inMetas = inputs.map((id) => this.tensorRegistry.get(id));
    const outMeta = this.tensorRegistry.get(output);
    if (inMetas.some((m) => !m) || !outMeta) {
      console.error(`Missing tensor metadata for op ${op}`);
      return;
    }
    const im = inMetas as TensorMetadata[];

    if (op === "SUM") {
      this.dispatchSum(im, outMeta);
      return;
    }
    if (op === "MATERIALIZE") {
      this.dispatchMaterialize(im, outMeta, params);
      return;
    }

    const pipeline = this.pipelines?.byOp.get(op);
    if (!pipeline) {
      // BMM reuses the MATMUL pipeline with per-batch offset uniforms.
      if (op === "BMM") {
        const mmPipe = this.pipelines?.byOp.get("MATMUL");
        if (!mmPipe) throw new Error(`WebGPU backend: MATMUL pipeline missing`);
        return this.dispatchBmm(mmPipe, im, outMeta, params);
      }
      throw new Error(`WebGPU backend: op '${op}' not yet ported`);
    }

    if (op === "MATMUL") return this.dispatchMatmul(pipeline, im, outMeta, params);
    if (op === "TRANSPOSE") return this.dispatchTranspose(pipeline, im, outMeta, params);
    if (op === "SOFTMAX" || op === "SOFTMAX_BACKWARD")
      return this.dispatchSoftmax(pipeline, op, im, outMeta, params);
    if (op === "RMS_NORM") return this.dispatchRmsNorm(pipeline, im, outMeta, params);
    if (op === "ROPE") return this.dispatchRope(pipeline, im, outMeta, params);
    if (op === "CAUSAL_SOFTMAX") return this.dispatchCausalSoftmax(pipeline, im, outMeta, params);
    if (op === "COPY_RANGE") return this.dispatchCopyRange(pipeline, im, outMeta, params);
    if (op === "FILL") return this.dispatchFill(pipeline, outMeta, params);
    if (op === "RANDN") return this.dispatchRandn(pipeline, outMeta);
    if (op === "EMBEDDING") return this.dispatchEmbedding(pipeline, im, outMeta, params);
    if (op === "EMBEDDING_BACKWARD")
      return this.dispatchEmbeddingBackward(pipeline, im, outMeta, params);
    if (op === "SUM_AXIS") return this.dispatchSumAxis(pipeline, im, outMeta, params);
    if (op === "CONV1D") return this.dispatchConv1d(pipeline, im, outMeta, params);
    if (op === "CONV_TRANSPOSE_1D") return this.dispatchConvTranspose1d(pipeline, im, outMeta, params);
    if (op === "CONCAT_SLAB") return this.dispatchConcatSlab(pipeline, im, outMeta, params);
    if (op === "LSTM_STEP") return this.dispatchLstmStep(pipeline, im, outMeta, params);
    if (op === "SNAKE_1D") return this.dispatchSnake1D(pipeline, im, outMeta, params);
    if (op === "STYLE_AFFINE") return this.dispatchStyleAffine(pipeline, im, outMeta, params);

    // Elementwise families: materialize any non-contiguous operand into scratch,
    // then dispatch the contiguous fast path.
    if (op === "ADD" || op === "SUB" || op === "MUL" || op === "DIV") {
      const inA = this.maybeMaterialize(im[0], params.shape, params.stridesA);
      const inB = this.maybeMaterialize(im[1], params.shape, params.stridesB);
      this.dispatchBinary(pipeline, inA.meta, inB.meta, outMeta);
      inA.free();
      inB.free();
      return;
    }
    if (op === "ADD_SCALAR_TENSOR") {
      return this.dispatchBinary(pipeline, im[0], im[1], outMeta);
    }
    if (op === "RELU_BACKWARD") {
      const inA = this.maybeMaterialize(im[0], params.shape, params.strides);
      this.dispatchBinary(pipeline, inA.meta, im[1], outMeta);
      inA.free();
      return;
    }
    if (op === "TANH_BACKWARD") {
      return this.dispatchBinary(pipeline, im[0], im[1], outMeta);
    }
    if (op === "GELU_BACKWARD") {
      const inA = this.maybeMaterialize(im[0], params.shape, params.strides);
      this.dispatchBinary(pipeline, inA.meta, im[1], outMeta);
      inA.free();
      return;
    }
    if (op === "SQRT_BACKWARD" || op === "RSQRT_BACKWARD" || op === "SIGMOID_BACKWARD") {
      return this.dispatchBinary(pipeline, im[0], im[1], outMeta);
    }
    if (op === "SILU_BACKWARD") {
      return this.dispatchBinary(pipeline, im[0], im[1], outMeta);
    }
    if (op === "LEAKY_RELU_BACKWARD") {
      return this.dispatchBinaryWithParam(pipeline, im[0], im[1], outMeta, required(params.negativeSlope, "negativeSlope"));
    }
    if (
      op === "NEG" ||
      op === "RELU" ||
      op === "EXP" ||
      op === "LOG" ||
      op === "TANH" ||
      op === "SIN" ||
      op === "COS" ||
      op === "GELU" ||
      op === "SQRT" ||
      op === "RSQRT" ||
      op === "SIGMOID" ||
      op === "SILU" ||
      op === "COPY"
    ) {
      const inA =
        op === "COPY" ? { meta: im[0], free: NOOP_FREE }
                      : this.maybeMaterialize(im[0], params.shape, params.strides);
      this.dispatchUnary(pipeline, inA.meta, outMeta);
      inA.free();
      return;
    }
    if (op === "LEAKY_RELU") {
      const inA = this.maybeMaterialize(im[0], params.shape, params.strides);
      this.dispatchUnaryWithParam(pipeline, inA.meta, outMeta, required(params.negativeSlope, "negativeSlope"));
      inA.free();
      return;
    }

    throw new Error(`WebGPU backend: op '${op}' dispatch missing`);
  }

  // Narrowed accessor for initialized WebGPU state. runOp() and init() are the
  // only entry points that can create work, so anything they call transitively
  // sees non-null fields; TypeScript can't see that, so we centralize the check.
  private requireReady() {
    if (
      !this.device ||
      !this.queue ||
      !this.uniforms ||
      !this.bindGroup ||
      !this.pipelines ||
      !this.allocator
    ) {
      throw new Error("WebGPU dispatcher not initialized");
    }
    return {
      device: this.device,
      queue: this.queue,
      uniforms: this.uniforms,
      bindGroup: this.bindGroup,
      pipelines: this.pipelines,
      allocator: this.allocator,
    };
  }

  private requirePipeline(op: string): GPUComputePipeline {
    const p = this.requireReady().pipelines.byOp.get(op);
    if (!p) throw new Error(`WebGPU pipeline for op '${op}' not registered`);
    return p;
  }

  // Returns the operand as-is if contiguous, otherwise a freshly materialized scratch
  // with a `free()` closure to return it to the allocator once the dispatch is encoded.
  private maybeMaterialize(
    src: TensorMetadata,
    shape: number[] | undefined,
    strides: number[] | undefined,
  ): { meta: TensorMetadata; free: () => void } {
    if (!shape || !strides || isContiguous(shape, strides)) {
      return { meta: src, free: NOOP_FREE };
    }
    const alloc = this.requireReady().allocator;
    const outSize = shape.reduce((a, b) => a * b, 1) * 4;
    const outOffset = alloc.allocate(outSize);
    this.encodeMaterialize(src, { offset: outOffset, size: outSize, isView: false }, shape, strides);
    return {
      meta: { offset: outOffset, size: outSize, isView: false },
      // Queue-ordered: subsequent allocs at this offset run after the encoded read.
      free: () => alloc.free(outOffset, outSize),
    };
  }

  private encodeMaterialize(
    src: TensorMetadata,
    out: TensorMetadata,
    shape: number[],
    strides: number[],
  ) {
    if (shape.length > 8) {
      throw new Error(`WebGPU MATERIALIZE supports at most 8 dims (got ${shape.length})`);
    }
    const pipeline = this.requirePipeline("MATERIALIZE");
    const count = out.size / 4;
    const u = new Uint32Array(20);
    u[0] = src.offset >>> 2;
    u[1] = out.offset >>> 2;
    u[2] = shape.length;
    u[3] = count;
    for (let i = 0; i < shape.length; i++) {
      u[4 + i] = shape[i];
      u[12 + i] = strides[i];
    }
    this.encodeAndSubmit(pipeline, u, Math.ceil(count / ELEMENTWISE_TILE), 1);
  }

  private dispatchMatmul(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    output: TensorMetadata,
    params: OpParams,
  ) {
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const k = required(params.k, "k");
    // The WebGPU matmul shader assumes contiguous row-major operands. Workers
    // and WASM matmul kernels accept strides directly; here we materialize.
    const inA = this.maybeMaterialize(inputs[0], [m, k], params.stridesA);
    const inB = this.maybeMaterialize(inputs[1], [k, n], params.stridesB);
    const u = new Uint32Array([
      inA.meta.offset >>> 2,
      inB.meta.offset >>> 2,
      output.offset >>> 2,
      m,
      n,
      k,
      0,
      m,
    ]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(m / MATMUL_TILE), Math.ceil(n / MATMUL_TILE));
    inA.free();
    inB.free();
  }

  // BMM reuses the MATMUL pipeline. One command encoding, but the matmul is
  // dispatched B times, each with input/output offsets shifted by the batch stride.
  private dispatchBmm(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    output: TensorMetadata,
    params: OpParams,
  ) {
    const batchCount = required(params.batchCount, "batchCount");
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const k = required(params.k, "k");
    const aOffF32 = inputs[0].offset >>> 2;
    const bOffF32 = inputs[1].offset >>> 2;
    const oOffF32 = output.offset >>> 2;
    for (let bi = 0; bi < batchCount; bi++) {
      const u = new Uint32Array([
        aOffF32 + bi * m * k,
        bOffF32 + bi * k * n,
        oOffF32 + bi * m * n,
        m,
        n,
        k,
        0,
        m,
      ]);
      this.encodeAndSubmit(pipeline, u, Math.ceil(m / MATMUL_TILE), Math.ceil(n / MATMUL_TILE));
    }
  }

  private dispatchBinary(
    pipeline: GPUComputePipeline,
    a: TensorMetadata,
    b: TensorMetadata,
    out: TensorMetadata,
  ) {
    const len = out.size / 4;
    const u = new Uint32Array([a.offset >>> 2, b.offset >>> 2, out.offset >>> 2, 0, len]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(len / ELEMENTWISE_TILE), 1);
  }

  private dispatchUnary(pipeline: GPUComputePipeline, input: TensorMetadata, out: TensorMetadata) {
    const len = out.size / 4;
    const u = new Uint32Array([input.offset >>> 2, out.offset >>> 2, 0, len]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(len / ELEMENTWISE_TILE), 1);
  }

  private dispatchUnaryWithParam(
    pipeline: GPUComputePipeline,
    input: TensorMetadata,
    out: TensorMetadata,
    param0: number,
  ) {
    const len = out.size / 4;
    const u = new Uint32Array(5);
    u[0] = input.offset >>> 2;
    u[1] = out.offset >>> 2;
    u[2] = 0;
    u[3] = len;
    new Float32Array(u.buffer)[4] = param0;
    this.encodeAndSubmit(pipeline, u, Math.ceil(len / ELEMENTWISE_TILE), 1);
  }

  private dispatchBinaryWithParam(
    pipeline: GPUComputePipeline,
    a: TensorMetadata,
    b: TensorMetadata,
    out: TensorMetadata,
    param0: number,
  ) {
    const len = out.size / 4;
    const u = new Uint32Array(6);
    u[0] = a.offset >>> 2;
    u[1] = b.offset >>> 2;
    u[2] = out.offset >>> 2;
    u[3] = 0;
    u[4] = len;
    new Float32Array(u.buffer)[5] = param0;
    this.encodeAndSubmit(pipeline, u, Math.ceil(len / ELEMENTWISE_TILE), 1);
  }

  private dispatchFill(pipeline: GPUComputePipeline, out: TensorMetadata, params: OpParams) {
    const len = out.size / 4;
    const val = required(params.value, "value");
    const valBits = new Uint32Array(new Float32Array([val]).buffer)[0];
    const u = new Uint32Array([out.offset >>> 2, 0, len, valBits]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(len / ELEMENTWISE_TILE), 1);
  }

  private dispatchRandn(pipeline: GPUComputePipeline, out: TensorMetadata) {
    const count = out.size / 4;
    const seed = (Date.now() & 0xffffffff) >>> 0 || 0x9e3779b9;
    const u = new Uint32Array([out.offset >>> 2, count, seed, 0]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(count / ELEMENTWISE_TILE), 1);
  }

  private dispatchTranspose(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const u = new Uint32Array([inputs[0].offset >>> 2, out.offset >>> 2, m, n]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(n / MATMUL_TILE), Math.ceil(m / MATMUL_TILE));
  }

  private dispatchSoftmax(
    pipeline: GPUComputePipeline,
    op: string,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    let p0: number, p1: number, p2: number;
    if (op === "SOFTMAX") {
      p0 = inputs[0].offset >>> 2;
      p1 = out.offset >>> 2;
      p2 = 0;
    } else {
      // SOFTMAX_BACKWARD: inputs[0]=output, inputs[1]=grad_output, output=grad_input
      p0 = inputs[0].offset >>> 2;
      p1 = inputs[1].offset >>> 2;
      p2 = out.offset >>> 2;
    }
    const u = new Uint32Array([p0, p1, p2, m, n, 0, m]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(m / ROW_TILE), 1);
  }

  private dispatchRmsNorm(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const eps = required(params.eps, "eps");
    const u = new Uint32Array(8);
    u[0] = inputs[0].offset >>> 2;
    u[1] = inputs[1].offset >>> 2;
    u[2] = out.offset >>> 2;
    u[3] = m;
    u[4] = n;
    new Float32Array(u.buffer)[5] = eps;
    u[6] = 0;
    u[7] = m;
    this.encodeAndSubmit(pipeline, u, Math.ceil(m / ROW_TILE), 1);
  }

  private dispatchRope(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const m = required(params.m, "m");
    const tSeq = required(params.tSeq, "tSeq");
    const dHalf = required(params.dHalf, "dHalf");
    const u = new Uint32Array([
      inputs[0].offset >>> 2,
      inputs[1].offset >>> 2,
      inputs[2].offset >>> 2,
      out.offset >>> 2,
      tSeq,
      dHalf,
      0,
      m,
    ]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(m / ROW_TILE), 1);
  }

  private dispatchCausalSoftmax(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const pastLen = params.pastLen ?? 0;
    const u = new Uint32Array([
      inputs[0].offset >>> 2,
      out.offset >>> 2,
      m,
      n,
      pastLen,
      0,
      m,
    ]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(m / ROW_TILE), 1);
  }

  private dispatchCopyRange(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const count = required(params.count, "count");
    const dstOffset = required(params.dstOffset, "dstOffset");
    const u = new Uint32Array([
      inputs[0].offset >>> 2,
      out.offset >>> 2,
      dstOffset,
      count,
    ]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(count / ELEMENTWISE_TILE), 1);
  }

  private dispatchMaterialize(inputs: TensorMetadata[], out: TensorMetadata, params: OpParams) {
    const shape = required(params.shape, "shape");
    const strides = required(params.strides, "strides");
    this.encodeMaterialize(inputs[0], out, shape, strides);
  }

  private dispatchSum(inputs: TensorMetadata[], out: TensorMetadata) {
    const { allocator: alloc } = this.requireReady();
    const partial = this.requirePipeline("SUM_PARTIAL");
    const final = this.requirePipeline("SUM_FINAL");

    const tempSize = SUM_PARTIALS * 4;
    const tempOffset = alloc.allocate(tempSize);

    // Phase 1: input -> temp[0..SUM_PARTIALS]
    const count = inputs[0].size / 4;
    const u1 = new Uint32Array([inputs[0].offset >>> 2, tempOffset >>> 2, count, SUM_PARTIALS]);
    this.encodeAndSubmit(partial, u1, Math.ceil(SUM_PARTIALS / ELEMENTWISE_TILE), 1);

    // Phase 2: temp -> output scalar
    const u2 = new Uint32Array([tempOffset >>> 2, out.offset >>> 2, 0, SUM_PARTIALS]);
    this.encodeAndSubmit(final, u2, 1, 1);

    // Free is safe: subsequent uses of tempOffset are queue-ordered after both dispatches.
    alloc.free(tempOffset, tempSize);
  }

  private dispatchSumAxis(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const shape = required(params.shape, "shape");
    const axis = required(params.axis, "axis");
    const axisSize = shape[axis];
    let innerSize = 1;
    for (let i = axis + 1; i < shape.length; i++) innerSize *= shape[i];
    const count = out.size / 4;
    // SumU layout: [input_off, output_off, count, num_partials, axis_size, inner_size]
    const u = new Uint32Array([
      inputs[0].offset >>> 2,
      out.offset >>> 2,
      count,
      0,
      axisSize,
      innerSize,
    ]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(count / 64), 1);
  }

  private dispatchConv1d(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const B = required(params.batchCount, "batchCount");
    const Cin = required(params.Cin, "Cin");
    const Lin = required(params.Lin, "Lin");
    const Cout = required(params.Cout, "Cout");
    const K = required(params.K, "K");
    const Lout = required(params.Lout, "Lout");
    const stride = required(params.stride, "stride");
    const padding = required(params.padding, "padding");
    const dilation = required(params.dilation, "dilation");
    const groups = params.groups ?? 1;
    const hasBias = !!params.hasBias;
    const u = new Uint32Array(15);
    u[0] = inputs[0].offset >>> 2;
    u[1] = inputs[1].offset >>> 2;
    u[2] = hasBias ? inputs[2].offset >>> 2 : 0;
    u[3] = out.offset >>> 2;
    u[4] = hasBias ? 1 : 0;
    u[5] = B;
    u[6] = Cin;
    u[7] = Lin;
    u[8] = Cout;
    u[9] = K;
    u[10] = Lout;
    const iview = new Int32Array(u.buffer);
    iview[11] = stride;
    iview[12] = padding;
    iview[13] = dilation;
    u[14] = groups;
    // 2D dispatch: (Lout_tiles, B*Cout). Each workgroup handles one (b, co) row
    // and cooperatively loads the weight[co, :, :] tile into shared memory.
    this.encodeAndSubmit(pipeline, u, Math.ceil(Lout / 256), B * Cout);
  }

  private dispatchConvTranspose1d(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const B = required(params.batchCount, "batchCount");
    const Cin = required(params.Cin, "Cin");
    const Lin = required(params.Lin, "Lin");
    const Cout = required(params.Cout, "Cout");
    const K = required(params.K, "K");
    const Lout = required(params.Lout, "Lout");
    const stride = required(params.stride, "stride");
    const padding = required(params.padding, "padding");
    const dilation = required(params.dilation, "dilation");
    const groups = params.groups ?? 1;
    const hasBias = !!params.hasBias;
    const u = new Uint32Array(15);
    u[0] = inputs[0].offset >>> 2;
    u[1] = inputs[1].offset >>> 2;
    u[2] = hasBias ? inputs[2].offset >>> 2 : 0;
    u[3] = out.offset >>> 2;
    u[4] = hasBias ? 1 : 0;
    u[5] = B;
    u[6] = Cin;
    u[7] = Lin;
    u[8] = Cout;
    u[9] = K;
    u[10] = Lout;
    const iview = new Int32Array(u.buffer);
    iview[11] = stride;
    iview[12] = padding;
    iview[13] = dilation;
    u[14] = groups;
    const total = B * Cout * Lout;
    this.encodeAndSubmit(pipeline, u, Math.ceil(total / 256), 1);
  }

  private dispatchLstmStep(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const hidden = required(params.hidden, "hidden");
    const inSize = required(params.inSize, "inSize");
    const batchSize = required(params.batchSize, "batchSize");
    const [x, h, c, wIh, wHh, bIh, bHh, cOut] = inputs;
    // Direct-write path: caller provides h_new relative element offset (into
    // the output tensor) and an 8th "input" tensor whose registry-tracked
    // offset is the c_new destination, plus its own relative element offset.
    // Fallback packed layout: h at out.offset, c at out.offset + B*H*4.
    const hNewOff = out.offset + (params.hNewOffElements ?? 0) * 4;
    const cNewOff = cOut
      ? cOut.offset + (params.cNewOffElements ?? 0) * 4
      : out.offset + batchSize * hidden * 4;
    const u = new Uint32Array([
      x.offset >>> 2,
      h.offset >>> 2,
      c.offset >>> 2,
      wIh.offset >>> 2,
      wHh.offset >>> 2,
      bIh.offset >>> 2,
      bHh.offset >>> 2,
      hNewOff >>> 2,
      cNewOff >>> 2,
      batchSize,
      hidden,
      inSize,
    ]);
    const total = batchSize * hidden;
    this.encodeAndSubmit(pipeline, u, Math.ceil(total / 64), 1);
  }

  private dispatchSnake1D(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const channels = required(params.axisSize, "axisSize");
    const inner = required(params.innerSize, "innerSize");
    const numel = out.size / 4;
    const [x, alpha] = inputs;
    const u = new Uint32Array([
      x.offset >>> 2,
      alpha.offset >>> 2,
      out.offset >>> 2,
      numel,
      channels,
      inner,
    ]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(numel / ELEMENTWISE_TILE), 1);
  }

  private dispatchStyleAffine(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const channels = required(params.axisSize, "axisSize");
    const inner = required(params.innerSize, "innerSize");
    const numel = out.size / 4;
    const [x, gamma, beta] = inputs;
    const u = new Uint32Array([
      x.offset >>> 2,
      gamma.offset >>> 2,
      beta.offset >>> 2,
      out.offset >>> 2,
      numel,
      channels,
      inner,
    ]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(numel / ELEMENTWISE_TILE), 1);
  }

  private dispatchConcatSlab(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const outerSize = required(params.outerSize, "outerSize");
    const inAxisSize = required(params.inAxisSize, "inAxisSize");
    const outAxisSize = required(params.outAxisSize, "outAxisSize");
    const axisOffset = required(params.axisOffset, "axisOffset");
    const innerSize = required(params.innerSize, "innerSize");
    const total = outerSize * inAxisSize * innerSize;
    const u = new Uint32Array([
      inputs[0].offset >>> 2,
      out.offset >>> 2,
      outerSize,
      inAxisSize,
      outAxisSize,
      axisOffset,
      innerSize,
      total,
    ]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(total / 256), 1);
  }

  private dispatchEmbedding(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const embeddingDim = required(params.embeddingDim, "embeddingDim");
    const count = out.size / 4;
    // EmbeddingU: [buf_w, buf_i, buf_o, embedding_dim, count, num_indices]
    const u = new Uint32Array([
      inputs[0].offset >>> 2,
      inputs[1].offset >>> 2,
      out.offset >>> 2,
      embeddingDim,
      count,
      0,
    ]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(count / ELEMENTWISE_TILE), 1);
  }

  private dispatchEmbeddingBackward(
    pipeline: GPUComputePipeline,
    inputs: TensorMetadata[],
    out: TensorMetadata,
    params: OpParams,
  ) {
    const embeddingDim = required(params.embeddingDim, "embeddingDim");
    const count = out.size / 4;
    const numIndices = inputs[0].size / 4;
    // Vocab-major: buf_w = weights_grad (out), buf_i = indices, buf_o = output_grad
    const u = new Uint32Array([
      out.offset >>> 2,
      inputs[0].offset >>> 2,
      inputs[1].offset >>> 2,
      embeddingDim,
      count,
      numIndices,
    ]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(count / 64), 1);
  }

  private encodeAndSubmit(
    pipeline: GPUComputePipeline,
    uniforms: Uint32Array,
    workgroupsX: number,
    workgroupsY: number,
  ) {
    const { device, queue, uniforms: uniformsBuf, bindGroup } = this.requireReady();
    queue.writeBuffer(
      uniformsBuf,
      0,
      uniforms.buffer as ArrayBuffer,
      uniforms.byteOffset,
      uniforms.byteLength,
    );
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY);
    pass.end();
    queue.submit([encoder.finish()]);
    this.inflight = queue.onSubmittedWorkDone();
  }

  set(tensorId: TensorId, offset: number, value: number): void {
    const meta = this.tensorRegistry.get(tensorId);
    if (!meta || !this.queue || !this.heap) return;
    const buf = new Float32Array([value]);
    this.queue.writeBuffer(this.heap, meta.offset + offset * 4, buf.buffer);
  }

  write(tensorId: TensorId, data: Float32Array): void {
    const meta = this.tensorRegistry.get(tensorId);
    if (!meta || !this.queue || !this.heap) return;
    this.queue.writeBuffer(
      this.heap,
      meta.offset,
      data.buffer as ArrayBuffer,
      data.byteOffset,
      data.byteLength,
    );
  }

  async read(tensorId: TensorId): Promise<Float32Array> {
    return this.readRegion(tensorId);
  }

  async readView(tensorId: TensorId): Promise<Float32Array> {
    return this.readRegion(tensorId);
  }

  async readValue(tensorId: TensorId, offset: number): Promise<number> {
    const arr = await this.readRegion(tensorId);
    return arr[offset];
  }

  private async readRegion(tensorId: TensorId): Promise<Float32Array> {
    const meta = this.tensorRegistry.get(tensorId);
    const device = this.device;
    const queue = this.queue;
    const heap = this.heap;
    if (!meta || !device || !queue || !heap) throw new Error("Dispatcher not initialized");

    await this.inflight;

    const staging = device.createBuffer({
      size: meta.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(heap, meta.offset, staging, 0, meta.size);
    queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const mapped = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return mapped;
  }
}

function required<T>(v: T | undefined, name: string): T {
  if (v === undefined) throw new Error(`Missing required param: ${name}`);
  return v;
}

function isContiguous(shape: number[] | undefined, strides: number[] | undefined): boolean {
  if (!shape || !strides) return true;
  let expected = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    if (shape[i] === 1) continue;
    if (strides[i] !== expected) return false;
    expected *= shape[i];
  }
  return true;
}
