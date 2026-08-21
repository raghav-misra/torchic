import type { Dispatcher } from "../dispatcher";
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

  async init(_threadCount = 0, memorySizeMB = 256): Promise<void> {
    if (this.device) return;

    const { device, queue } = await requestContext();
    this.device = device;
    this.queue = queue;

    const heapBytes = memorySizeMB * 1024 * 1024;
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
    if (!pipeline) throw new Error(`WebGPU backend: op '${op}' not yet ported`);

    if (op === "MATMUL") return this.dispatchMatmul(pipeline, im, outMeta, params);
    if (op === "TRANSPOSE") return this.dispatchTranspose(pipeline, im, outMeta, params);
    if (op === "SOFTMAX" || op === "SOFTMAX_BACKWARD")
      return this.dispatchSoftmax(pipeline, op, im, outMeta, params);
    if (op === "FILL") return this.dispatchFill(pipeline, outMeta, params);
    if (op === "RANDN") return this.dispatchRandn(pipeline, outMeta);

    // Elementwise families: contiguous fast path only.
    if (op === "ADD" || op === "SUB" || op === "MUL" || op === "DIV") {
      this.assertContigBinary(params, op);
      return this.dispatchBinary(pipeline, im[0], im[1], outMeta);
    }
    if (op === "ADD_SCALAR_TENSOR") {
      return this.dispatchBinary(pipeline, im[0], im[1], outMeta);
    }
    if (op === "RELU_BACKWARD") {
      this.assertContigUnary(params, op);
      return this.dispatchBinary(pipeline, im[0], im[1], outMeta);
    }
    if (op === "TANH_BACKWARD") {
      return this.dispatchBinary(pipeline, im[0], im[1], outMeta);
    }
    if (
      op === "NEG" ||
      op === "RELU" ||
      op === "EXP" ||
      op === "LOG" ||
      op === "TANH" ||
      op === "COPY"
    ) {
      if (op !== "COPY") this.assertContigUnary(params, op);
      return this.dispatchUnary(pipeline, im[0], outMeta);
    }

    throw new Error(`WebGPU backend: op '${op}' dispatch missing`);
  }

  private assertContigBinary(params: OpParams, op: string) {
    if (!isContiguous(params.shape, params.stridesA) || !isContiguous(params.shape, params.stridesB)) {
      throw new Error(`WebGPU backend: broadcasted ${op} not yet ported`);
    }
  }
  private assertContigUnary(params: OpParams, op: string) {
    if (!isContiguous(params.shape, params.strides)) {
      throw new Error(`WebGPU backend: strided ${op} not yet ported`);
    }
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
    const u = new Uint32Array([
      inputs[0].offset >>> 2,
      inputs[1].offset >>> 2,
      output.offset >>> 2,
      m,
      n,
      k,
      0,
      m,
    ]);
    this.encodeAndSubmit(pipeline, u, Math.ceil(m / MATMUL_TILE), Math.ceil(n / MATMUL_TILE));
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

  private dispatchMaterialize(inputs: TensorMetadata[], out: TensorMetadata, params: OpParams) {
    const pipeline = this.pipelines!.byOp.get("MATERIALIZE")!;
    const shape = required(params.shape, "shape");
    const strides = required(params.strides, "strides");
    if (shape.length > 8) {
      throw new Error(`WebGPU MATERIALIZE supports at most 8 dims (got ${shape.length})`);
    }
    const count = out.size / 4;
    // Uniform layout: [in_off, out_off, ndim, count, shape[0..8], strides[0..8]] = 20 u32
    const u = new Uint32Array(20);
    u[0] = inputs[0].offset >>> 2;
    u[1] = out.offset >>> 2;
    u[2] = shape.length;
    u[3] = count;
    for (let i = 0; i < shape.length; i++) {
      u[4 + i] = shape[i];
      u[12 + i] = strides[i];
    }
    this.encodeAndSubmit(pipeline, u, Math.ceil(count / ELEMENTWISE_TILE), 1);
  }

  private dispatchSum(inputs: TensorMetadata[], out: TensorMetadata) {
    const alloc = this.allocator!;
    const partial = this.pipelines!.byOp.get("SUM_PARTIAL")!;
    const final = this.pipelines!.byOp.get("SUM_FINAL")!;

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

  private encodeAndSubmit(
    pipeline: GPUComputePipeline,
    uniforms: Uint32Array,
    workgroupsX: number,
    workgroupsY: number,
  ) {
    const device = this.device!;
    const queue = this.queue!;
    queue.writeBuffer(
      this.uniforms!,
      0,
      uniforms.buffer as ArrayBuffer,
      uniforms.byteOffset,
      uniforms.byteLength,
    );
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.bindGroup!);
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
    if (strides[i] !== expected) return false;
    expected *= shape[i];
  }
  return true;
}
