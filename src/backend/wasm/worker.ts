import { MemoryAllocator } from "../memory";
import { defineWorkerOnMessage } from "../../shared/utils";
import type { OpParams, BufferRegion } from "../../shared/types";
import { ComputeResponse, TypedPort } from "../../shared/types";
import type { WasmCoordinatorRequest, WasmComputeRequest } from "./types";
import { instantiateKernels, type WasmInstance } from "./loader";

type WorkerRole = "COORDINATOR" | "COMPUTE";

function required<T>(val: T | undefined, name: string): T {
  if (val === undefined) throw new Error(`Missing required param: ${name}`);
  return val;
}

interface TensorMetadata {
  offset: number;
  size: number;
  isView: boolean;
}

let role: WorkerRole = "COORDINATOR";
let memoryAllocator: MemoryAllocator | null = null;
let buffer: SharedArrayBuffer | null = null;
let wasm: WasmInstance | null = null;
const tensorRegistry = new Map<string, TensorMetadata>();

const computePorts: TypedPort<WasmComputeRequest, ComputeResponse>[] = [];
const pendingTasks = new Map<string, { resolve: () => void; count: number }>();
let commandQueue = Promise.resolve();

self.onmessage = defineWorkerOnMessage<WasmCoordinatorRequest | WasmComputeRequest>(
  (data, ports) => {
    const { type } = data;

    if (type === "INIT_WASM_COORDINATOR") {
      role = "COORDINATOR";
      const payload = data.payload;
      buffer = payload.memory.buffer as unknown as SharedArrayBuffer;
      memoryAllocator = new MemoryAllocator(buffer, payload.heapBase);
      self.postMessage({ id: data.id, data: { status: "ok" } });
      return;
    }

    if (type === "INIT_WASM_WORKER") {
      role = "COMPUTE";
      const payload = data.payload;
      const port = ports[0];
      const coordinatorPort = new TypedPort<ComputeResponse, WasmComputeRequest>(port);
      // Instantiation is async but message dispatch is sync; queue kernel handling
      // behind the ready promise so early EXECUTE_TASK never races the instance.
      const ready = instantiateKernels(payload.module, payload.memory).then((w) => {
        wasm = w;
      });
      setupComputeWorker(coordinatorPort, ready);
      return;
    }

    if (type === "ADD_WORKER") {
      if (role !== "COORDINATOR") return;
      const port = ports[0];
      const typedPort = new TypedPort<WasmComputeRequest, ComputeResponse>(port);
      computePorts.push(typedPort);
      setupCoordinatorPort(typedPort);
      return;
    }

    if (role === "COORDINATOR") {
      const req = data as WasmCoordinatorRequest;

      commandQueue = commandQueue
        .then(async () => {
          switch (req.type) {
            case "ALLOC":
              handleAlloc(req.payload);
              break;
            case "ALLOC_VIEW":
              handleAllocView(req.payload);
              break;
            case "FREE":
              handleFree(req.payload);
              break;
            case "SET":
              handleSet(req.payload);
              break;
            case "WRITE":
              handleWrite(req.payload);
              break;
            case "OP":
              await handleOp({ ...req.payload, params: req.payload.params || {} }, req.id);
              break;
            case "READ":
              handleRead(req.payload, req.id);
              break;
            case "READ_VIEW":
              handleReadView(req.payload, req.id);
              break;
            case "READ_VALUE":
              handleReadValue(req.payload, req.id);
              break;
          }
        })
        .catch((err) => {
          console.error("WASM Coordinator Error:", err);
          const reqId = "id" in req ? (req as { id: string }).id : undefined;
          if (reqId) {
            self.postMessage({ id: reqId, error: err.message });
          }
        });
    }
  },
);

function setupCoordinatorPort(port: TypedPort<WasmComputeRequest, ComputeResponse>) {
  port.onMessage((data) => {
    if (data.type === "TASK_DONE") {
      const task = pendingTasks.get(data.taskId);
      if (task) {
        task.count--;
        if (task.count === 0) {
          task.resolve();
          pendingTasks.delete(data.taskId);
        }
      }
    }
  });
}

function handleAlloc(payload: { id: string; size: number }) {
  if (!memoryAllocator) return;
  try {
    const offset = memoryAllocator.allocate(payload.size);
    tensorRegistry.set(payload.id, { offset, size: payload.size, isView: false });
  } catch (e: unknown) {
    console.error("Allocation failed:", e instanceof Error ? e.message : e);
  }
}

function handleAllocView(payload: { id: string; parentId: string; offset?: number }) {
  const parentMeta = tensorRegistry.get(payload.parentId);
  if (!parentMeta) {
    console.error(`Cannot create view: parent tensor ${payload.parentId} not found`);
    return;
  }
  const providedOffset = payload.offset;
  const offset =
    typeof providedOffset === "number" ? parentMeta.offset + providedOffset : parentMeta.offset;
  tensorRegistry.set(payload.id, { offset, size: parentMeta.size, isView: true });
}

function handleFree(payload: { id: string }) {
  if (!memoryAllocator) return;
  const meta = tensorRegistry.get(payload.id);
  if (!meta) return;
  if (!meta.isView) memoryAllocator.free(meta.offset, meta.size);
  tensorRegistry.delete(payload.id);
}

function handleSet(payload: { id: string; offset: number; value: number }) {
  if (!buffer) return;
  const meta = tensorRegistry.get(payload.id);
  if (!meta) return;
  const view = new Float32Array(buffer, meta.offset, meta.size / 4);
  view[payload.offset] = payload.value;
}

function handleWrite(payload: { id: string; data: Float32Array }) {
  if (!buffer) return;
  const meta = tensorRegistry.get(payload.id);
  if (!meta) return;
  const view = new Float32Array(buffer, meta.offset, meta.size / 4);
  view.set(payload.data);
}

async function handleOp(
  payload: { op: string; inputs: string[]; output: string; params: OpParams },
  reqId?: string,
) {
  const inputMetas = payload.inputs.map((id) => tensorRegistry.get(id));
  const outputMeta = tensorRegistry.get(payload.output);

  if (inputMetas.some((m) => !m) || !outputMeta) {
    console.error("Missing tensor metadata for op:", payload.op);
    return;
  }

  const inputs = inputMetas as TensorMetadata[];
  const output = outputMeta;
  const numWorkers = computePorts.length;

  // SUM: two-phase reduce (partial sums per worker -> final scalar).
  if (payload.op === "SUM") {
    if (!memoryAllocator) return;

    const tempSize = numWorkers * 4;
    const tempOffset = memoryAllocator.allocate(tempSize);

    const taskId1 = crypto.randomUUID();
    const done1 = new Promise<void>((resolve) => {
      pendingTasks.set(taskId1, { resolve, count: numWorkers });
    });
    computePorts.forEach((port, index) => {
      port.postMessage({
        type: "EXECUTE_TASK",
        taskId: taskId1,
        op: "SUM_PARTIAL",
        inputs: [{ offset: inputs[0].offset, size: inputs[0].size }],
        output: { offset: tempOffset, size: tempSize },
        params: { outIndex: index },
        workerIndex: index,
        totalWorkers: numWorkers,
      });
    });
    await done1;

    const taskId2 = crypto.randomUUID();
    const done2 = new Promise<void>((resolve) => {
      pendingTasks.set(taskId2, { resolve, count: 1 });
    });
    computePorts[0].postMessage({
      type: "EXECUTE_TASK",
      taskId: taskId2,
      op: "SUM_FINAL",
      inputs: [{ offset: tempOffset, size: tempSize }],
      output: { offset: output.offset, size: output.size },
      params: { n: numWorkers },
      workerIndex: 0,
      totalWorkers: 1,
    });
    await done2;

    memoryAllocator.free(tempOffset, tempSize);
    if (reqId) self.postMessage({ id: reqId, data: { status: "done" } });
    return;
  }

  // Single-worker: scatter-add races across workers otherwise.
  if (payload.op === "EMBEDDING_BACKWARD") {
    const taskId = crypto.randomUUID();
    const done = new Promise<void>((resolve) => {
      pendingTasks.set(taskId, { resolve, count: 1 });
    });
    computePorts[0].postMessage({
      type: "EXECUTE_TASK",
      taskId,
      op: payload.op,
      inputs: inputs.map((m) => ({ offset: m.offset, size: m.size })),
      output: { offset: output.offset, size: output.size },
      params: payload.params,
      workerIndex: 0,
      totalWorkers: 1,
    });
    await done;
    if (reqId) self.postMessage({ id: reqId, data: { status: "done" } });
    return;
  }

  // Needs shape/strides visible to kernels; pack them into a shared scratch region.
  if (payload.op === "MATERIALIZE") {
    if (!memoryAllocator || !buffer) return;
    const shape = required(payload.params.shape, "shape");
    const strides = required(payload.params.strides, "strides");
    await runMaterialize(inputs[0].offset, inputs[0].size, output.offset, output.size, shape, strides);
    if (reqId) self.postMessage({ id: reqId, data: { status: "done" } });
    return;
  }

  // Matmul needs a per-worker private scratch panel for A-packing. Every worker
  // instance's wasm __stack_pointer starts at the same value in shared memory,
  // so we can't stack-alloc inside the kernel without workers racing.
  if (payload.op === "MATMUL" || payload.op === "BMM") {
    if (!memoryAllocator) return;
    const SCRATCH_BYTES_PER_WORKER = 4 * 256 * 4; // MR(4) * KC(256) * sizeof(f32)
    const scratchSize = numWorkers * SCRATCH_BYTES_PER_WORKER;
    const scratchOffset = memoryAllocator.allocate(scratchSize);

    const taskId = crypto.randomUUID();
    const done = new Promise<void>((resolve) => {
      pendingTasks.set(taskId, { resolve, count: numWorkers });
    });
    computePorts.forEach((port, index) => {
      port.postMessage({
        type: "EXECUTE_TASK",
        taskId,
        op: payload.op,
        inputs: inputs.map((m) => ({ offset: m.offset, size: m.size })),
        output: { offset: output.offset, size: output.size },
        params: {
          ...payload.params,
          scratchPtr: scratchOffset + index * SCRATCH_BYTES_PER_WORKER,
        },
        workerIndex: index,
        totalWorkers: numWorkers,
      });
    });
    await done;
    memoryAllocator.free(scratchOffset, scratchSize);
    if (reqId) self.postMessage({ id: reqId, data: { status: "done" } });
    return;
  }

  // Elementwise family: uniform chunk-slicing, contiguous-only fast path.
  if (isBinaryElementwise(payload.op) || isStridedUnary(payload.op)) {
    await runElementwiseWithMaterialize(payload.op, inputs, output, payload.params);
    if (reqId) self.postMessage({ id: reqId, data: { status: "done" } });
    return;
  }

  if (payload.op === "SUM_AXIS") {
    const shape = required(payload.params.shape, "shape");
    const axis = required(payload.params.axis, "axis");
    const axisSize = shape[axis];
    const innerSize = shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    await fanoutToWorkers("SUM_AXIS", inputs, output, { axisSize, innerSize });
    if (reqId) self.postMessage({ id: reqId, data: { status: "done" } });
    return;
  }

  await fanoutToWorkers(payload.op, inputs, output, payload.params);
  if (reqId) self.postMessage({ id: reqId, data: { status: "done" } });
}

const BINARY_ELEMENTWISE = new Set([
  "ADD",
  "SUB",
  "MUL",
  "DIV",
  "RELU_BACKWARD",
  "TANH_BACKWARD",
  "GELU_BACKWARD",
  "SQRT_BACKWARD",
  "RSQRT_BACKWARD",
  "SIGMOID_BACKWARD",
  "LEAKY_RELU_BACKWARD",
  "SILU_BACKWARD",
  "ADD_SCALAR_TENSOR",
]);

const STRIDED_UNARY = new Set([
  "NEG",
  "RELU",
  "EXP",
  "LOG",
  "TANH",
  "SIN",
  "COS",
  "GELU",
  "SQRT",
  "RSQRT",
  "SIGMOID",
  "LEAKY_RELU",
  "SILU",
  "COPY",
]);

function isBinaryElementwise(op: string): boolean {
  return BINARY_ELEMENTWISE.has(op);
}

function isStridedUnary(op: string): boolean {
  return STRIDED_UNARY.has(op);
}

async function fanoutToWorkers(
  op: string,
  inputs: TensorMetadata[],
  output: TensorMetadata,
  params: OpParams,
) {
  const numWorkers = computePorts.length;
  const taskId = crypto.randomUUID();
  const done = new Promise<void>((resolve) => {
    pendingTasks.set(taskId, { resolve, count: numWorkers });
  });
  computePorts.forEach((port, index) => {
    port.postMessage({
      type: "EXECUTE_TASK",
      taskId,
      op,
      inputs: inputs.map((m) => ({ offset: m.offset, size: m.size })),
      output: { offset: output.offset, size: output.size },
      params,
      workerIndex: index,
      totalWorkers: numWorkers,
    });
  });
  await done;
}

async function runMaterialize(
  srcOffset: number,
  srcSize: number,
  dstOffset: number,
  dstSize: number,
  shape: number[],
  strides: number[],
) {
  if (!memoryAllocator || !buffer) throw new Error("WASM coordinator not initialized");
  const ndim = shape.length;
  const scratchSize = ndim * 8;
  const scratchOffset = memoryAllocator.allocate(scratchSize);
  const view = new Uint32Array(buffer, scratchOffset, ndim * 2);
  for (let i = 0; i < ndim; i++) {
    view[i] = shape[i];
    view[ndim + i] = strides[i];
  }
  await fanoutToWorkers(
    "MATERIALIZE",
    [{ offset: srcOffset, size: srcSize, isView: false }],
    { offset: dstOffset, size: dstSize, isView: false },
    { ndim, shapePtr: scratchOffset, stridesPtr: scratchOffset + ndim * 4 },
  );
  memoryAllocator.free(scratchOffset, scratchSize);
}

async function materializeToScratch(
  src: TensorMetadata,
  shape: number[],
  strides: number[],
): Promise<TensorMetadata> {
  if (!memoryAllocator) throw new Error("WASM coordinator not initialized");
  const size = shape.reduce((a, b) => a * b, 1) * 4;
  const offset = memoryAllocator.allocate(size);
  await runMaterialize(src.offset, src.size, offset, size, shape, strides);
  return { offset, size, isView: false };
}

// Materialize non-contiguous operands into scratch, then run the contiguous fast path.
async function runElementwiseWithMaterialize(
  op: string,
  inputs: TensorMetadata[],
  output: TensorMetadata,
  params: OpParams,
) {
  const scratches: TensorMetadata[] = [];
  try {
    const workInputs = [...inputs];

    if (isBinaryElementwise(op)) {
      const shape = params.shape;
      if (shape && params.stridesA && !isContiguous(shape, params.stridesA)) {
        const mat = await materializeToScratch(inputs[0], shape, params.stridesA);
        workInputs[0] = mat;
        scratches.push(mat);
      }
      if (shape && params.stridesB && !isContiguous(shape, params.stridesB)) {
        const mat = await materializeToScratch(inputs[1], shape, params.stridesB);
        workInputs[1] = mat;
        scratches.push(mat);
      }
    } else if (isStridedUnary(op)) {
      const shape = params.shape;
      if (shape && params.strides && !isContiguous(shape, params.strides)) {
        const mat = await materializeToScratch(inputs[0], shape, params.strides);
        workInputs[0] = mat;
        scratches.push(mat);
      }
    }

    // Strip stride params so kernels take the contig fast path.
    const cleanParams: OpParams = { ...params };
    delete cleanParams.shape;
    delete cleanParams.strides;
    delete cleanParams.stridesA;
    delete cleanParams.stridesB;

    await fanoutToWorkers(op, workInputs, output, cleanParams);
  } finally {
    for (const s of scratches) memoryAllocator?.free(s.offset, s.size);
  }
}

function handleRead(payload: { id: string }, reqId: string) {
  const meta = tensorRegistry.get(payload.id);
  if (!meta || !buffer) return;
  const src = new Float32Array(buffer, meta.offset, meta.size / 4);
  const copy = new Float32Array(src);
  self.postMessage({ id: reqId, data: { data: copy } }, [copy.buffer]);
}

function handleReadView(payload: { id: string }, reqId: string) {
  const meta = tensorRegistry.get(payload.id);
  if (!meta) {
    self.postMessage({ id: reqId, error: `Tensor ${payload.id} not found` });
    return;
  }
  self.postMessage({ id: reqId, data: { offset: meta.offset, size: meta.size } });
}

function handleReadValue(payload: { id: string; offset: number }, reqId: string) {
  const meta = tensorRegistry.get(payload.id);
  if (!meta || !buffer) return;
  const view = new Float32Array(buffer, meta.offset, meta.size / 4);
  self.postMessage({ id: reqId, data: { value: view[payload.offset] } });
}

function setupComputeWorker(
  port: TypedPort<ComputeResponse, WasmComputeRequest>,
  ready: Promise<void>,
) {
  let queue = ready;
  port.onMessage((data) => {
    if (data.type !== "EXECUTE_TASK") return;
    queue = queue.then(() => {
      executeKernel(
        data.op,
        data.inputs,
        data.output,
        data.params,
        data.workerIndex,
        data.totalWorkers,
      );
      port.postMessage({ type: "TASK_DONE", taskId: data.taskId });
    });
  });
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

function executeKernel(
  op: string,
  inputs: BufferRegion[],
  output: BufferRegion,
  params: OpParams,
  workerIndex: number,
  totalWorkers: number,
) {
  if (!wasm) throw new Error("WASM instance not ready");
  const exports = wasm.exports;

  if (op === "MATMUL") {
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const k = required(params.k, "k");
    const rowsPerWorker = Math.ceil(m / totalWorkers);
    const startRow = workerIndex * rowsPerWorker;
    const endRow = Math.min(startRow + rowsPerWorker, m);
    if (startRow >= m) return;

    const stridesA = params.stridesA;
    const stridesB = params.stridesB;
    const aRowStride = stridesA ? stridesA[0] : k;
    const aColStride = stridesA ? stridesA[1] : 1;
    const bRowStride = stridesB ? stridesB[0] : n;
    const bColStride = stridesB ? stridesB[1] : 1;

    exports.matmul(
      inputs[0].offset,
      inputs[1].offset,
      output.offset,
      m,
      n,
      k,
      startRow,
      endRow,
      aRowStride,
      aColStride,
      bRowStride,
      bColStride,
      required(params.scratchPtr, "scratchPtr"),
    );
    return;
  }

  if (op === "BMM") {
    const batchCount = required(params.batchCount, "batchCount");
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const k = required(params.k, "k");
    const perWorker = Math.ceil(batchCount / totalWorkers);
    const startBatch = workerIndex * perWorker;
    const endBatch = Math.min(startBatch + perWorker, batchCount);
    if (startBatch >= batchCount) return;

    exports.bmm(
      inputs[0].offset,
      inputs[1].offset,
      output.offset,
      batchCount,
      m,
      n,
      k,
      startBatch,
      endBatch,
      required(params.scratchPtr, "scratchPtr"),
    );
    return;
  }

  if (op === "CONV1D" || op === "CONV_TRANSPOSE_1D") {
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
    const biasOffset = hasBias ? inputs[2].offset : 0;
    const perWorker = Math.ceil(B / totalWorkers);
    const startBatch = workerIndex * perWorker;
    const endBatch = Math.min(startBatch + perWorker, B);
    if (startBatch >= B) return;

    const fn = op === "CONV1D" ? exports.conv1d : exports.conv_transpose1d;
    fn(
      inputs[0].offset,
      inputs[1].offset,
      biasOffset,
      output.offset,
      hasBias ? 1 : 0,
      B,
      Cin,
      Lin,
      Cout,
      K,
      Lout,
      stride,
      padding,
      dilation,
      groups,
      startBatch,
      endBatch,
    );
    return;
  }

  if (op === "CONCAT_SLAB") {
    const outerSize = required(params.outerSize, "outerSize");
    const inAxisSize = required(params.inAxisSize, "inAxisSize");
    const outAxisSize = required(params.outAxisSize, "outAxisSize");
    const axisOffset = required(params.axisOffset, "axisOffset");
    const innerSize = required(params.innerSize, "innerSize");
    const total = outerSize * inAxisSize * innerSize;
    const perWorker = Math.ceil(total / totalWorkers);
    const start = workerIndex * perWorker;
    const end = Math.min(start + perWorker, total);
    if (start >= total) return;
    exports.concat_slab(
      inputs[0].offset,
      output.offset,
      outerSize,
      inAxisSize,
      outAxisSize,
      axisOffset,
      innerSize,
      start,
      end,
    );
    return;
  }

  if (op === "LSTM_STEP") {
    if (workerIndex !== 0) return;
    const hidden = required(params.hidden, "hidden");
    const inSize = required(params.inSize, "inSize");
    const batchSize = required(params.batchSize, "batchSize");
    const hOutPtr = params.hNewOffBytes ?? output.offset;
    const cOutPtr = params.cNewOffBytes ?? (output.offset + batchSize * hidden * 4);
    exports.lstm_step(
      inputs[0].offset, inputs[1].offset, inputs[2].offset,
      inputs[3].offset, inputs[4].offset, inputs[5].offset, inputs[6].offset,
      hOutPtr, cOutPtr, batchSize, hidden, inSize,
    );
    return;
  }

  if (op === "SOFTMAX" || op === "SOFTMAX_BACKWARD" || op === "TRANSPOSE") {
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const rowsTotal = op === "TRANSPOSE" ? n : m;
    const rowsPerWorker = Math.ceil(rowsTotal / totalWorkers);
    const startRow = workerIndex * rowsPerWorker;
    const endRow = Math.min(startRow + rowsPerWorker, rowsTotal);
    if (startRow >= rowsTotal) return;

    if (op === "SOFTMAX") {
      exports.softmax2d(inputs[0].offset, output.offset, m, n, startRow, endRow);
    } else if (op === "SOFTMAX_BACKWARD") {
      exports.softmax_backward2d(
        inputs[0].offset,
        inputs[1].offset,
        output.offset,
        m,
        n,
        startRow,
        endRow,
      );
    } else {
      exports.transpose(inputs[0].offset, output.offset, m, n, startRow, endRow);
    }
    return;
  }

  if (op === "SUM_PARTIAL") {
    const total = inputs[0].size / 4;
    const chunk = Math.ceil(total / totalWorkers);
    const start = workerIndex * chunk;
    const end = Math.min(start + chunk, total);
    if (start >= total) {
      // Ensure this worker's slot is zeroed even when it gets no work.
      exports.fill(output.offset, 0, required(params.outIndex, "outIndex"), required(params.outIndex, "outIndex") + 1);
      return;
    }
    exports.sum_partial(
      inputs[0].offset,
      output.offset,
      required(params.outIndex, "outIndex"),
      start,
      end,
    );
    return;
  }

  if (op === "SUM_FINAL") {
    exports.sum_final(inputs[0].offset, output.offset, required(params.n, "n"));
    return;
  }

  if (op === "EMBEDDING") {
    const total = output.size / 4;
    const chunk = Math.ceil(total / totalWorkers);
    const start = workerIndex * chunk;
    const end = Math.min(start + chunk, total);
    if (start >= total) return;
    exports.embedding(
      inputs[0].offset,
      inputs[1].offset,
      output.offset,
      required(params.embeddingDim, "embeddingDim"),
      start,
      end,
    );
    return;
  }

  if (op === "EMBEDDING_BACKWARD") {
    // Coordinator dispatched this on a single worker; grad_output length lives in inputs[1].
    const total = inputs[1].size / 4;
    exports.embedding_backward(
      output.offset,
      inputs[0].offset,
      inputs[1].offset,
      required(params.embeddingDim, "embeddingDim"),
      0,
      total,
    );
    return;
  }

  // Elementwise family: uniform chunk-slicing, contiguous-only fast path.
  const totalElements = output.size / 4;
  const chunkSize = Math.ceil(totalElements / totalWorkers);
  const start = workerIndex * chunkSize;
  const end = Math.min(start + chunkSize, totalElements);
  if (start >= totalElements) return;

  const stridedBinary = () => {
    if (!isContiguous(params.shape, params.stridesA) || !isContiguous(params.shape, params.stridesB)) {
      throw new Error(`WASM backend: broadcasted ${op} not yet ported`);
    }
  };
  const stridedUnary = () => {
    if (!isContiguous(params.shape, params.strides)) {
      throw new Error(`WASM backend: strided ${op} not yet ported`);
    }
  };

  switch (op) {
    case "ADD":
      stridedBinary();
      exports.add(inputs[0].offset, inputs[1].offset, output.offset, start, end);
      return;
    case "SUB":
      stridedBinary();
      exports.sub(inputs[0].offset, inputs[1].offset, output.offset, start, end);
      return;
    case "MUL":
      stridedBinary();
      exports.mul(inputs[0].offset, inputs[1].offset, output.offset, start, end);
      return;
    case "DIV":
      stridedBinary();
      exports.div(inputs[0].offset, inputs[1].offset, output.offset, start, end);
      return;
    case "NEG":
      stridedUnary();
      exports.neg(inputs[0].offset, output.offset, start, end);
      return;
    case "RELU":
      stridedUnary();
      exports.relu(inputs[0].offset, output.offset, start, end);
      return;
    case "RELU_BACKWARD":
      stridedUnary();
      exports.relu_backward(
        inputs[0].offset,
        inputs[1].offset,
        output.offset,
        start,
        end,
      );
      return;
    case "EXP":
      stridedUnary();
      exports.exp(inputs[0].offset, output.offset, start, end);
      return;
    case "LOG":
      stridedUnary();
      exports.log(inputs[0].offset, output.offset, start, end);
      return;
    case "TANH":
      stridedUnary();
      exports.tanh(inputs[0].offset, output.offset, start, end);
      return;
    case "TANH_BACKWARD":
      exports.tanh_backward(
        inputs[0].offset,
        inputs[1].offset,
        output.offset,
        start,
        end,
      );
      return;
    case "SIN":
      stridedUnary();
      exports.sin(inputs[0].offset, output.offset, start, end);
      return;
    case "COS":
      stridedUnary();
      exports.cos(inputs[0].offset, output.offset, start, end);
      return;
    case "GELU":
      stridedUnary();
      exports.gelu(inputs[0].offset, output.offset, start, end);
      return;
    case "GELU_BACKWARD":
      exports.gelu_backward(
        inputs[0].offset,
        inputs[1].offset,
        output.offset,
        start,
        end,
      );
      return;
    case "SQRT":
      stridedUnary();
      exports.sqrt_op(inputs[0].offset, output.offset, start, end);
      return;
    case "SQRT_BACKWARD":
      exports.sqrt_backward(
        inputs[0].offset,
        inputs[1].offset,
        output.offset,
        start,
        end,
      );
      return;
    case "RSQRT":
      stridedUnary();
      exports.rsqrt_op(inputs[0].offset, output.offset, start, end);
      return;
    case "RSQRT_BACKWARD":
      exports.rsqrt_backward(
        inputs[0].offset,
        inputs[1].offset,
        output.offset,
        start,
        end,
      );
      return;
    case "SIGMOID":
      stridedUnary();
      exports.sigmoid(inputs[0].offset, output.offset, start, end);
      return;
    case "SIGMOID_BACKWARD":
      exports.sigmoid_backward(
        inputs[0].offset,
        inputs[1].offset,
        output.offset,
        start,
        end,
      );
      return;
    case "LEAKY_RELU":
      stridedUnary();
      exports.leaky_relu(
        inputs[0].offset,
        output.offset,
        required(params.negativeSlope, "negativeSlope"),
        start,
        end,
      );
      return;
    case "LEAKY_RELU_BACKWARD":
      exports.leaky_relu_backward(
        inputs[0].offset,
        inputs[1].offset,
        output.offset,
        required(params.negativeSlope, "negativeSlope"),
        start,
        end,
      );
      return;
    case "SILU":
      stridedUnary();
      exports.silu(inputs[0].offset, output.offset, start, end);
      return;
    case "SILU_BACKWARD":
      exports.silu_backward(
        inputs[0].offset,
        inputs[1].offset,
        output.offset,
        start,
        end,
      );
      return;
    case "FILL":
      exports.fill(output.offset, required(params.value, "value"), start, end);
      return;
    case "COPY":
      exports.copy(inputs[0].offset, output.offset, start, end);
      return;
    case "MATERIALIZE":
      exports.materialize(
        inputs[0].offset,
        output.offset,
        start,
        end,
        required(params.ndim, "ndim"),
        required(params.shapePtr, "shapePtr"),
        required(params.stridesPtr, "stridesPtr"),
      );
      return;
    case "SUM_AXIS":
      exports.sum_axis(
        inputs[0].offset,
        output.offset,
        required(params.axisSize, "axisSize"),
        required(params.innerSize, "innerSize"),
        start,
        end,
      );
      return;
    case "ADD_SCALAR_TENSOR":
      exports.add_scalar_tensor(
        inputs[0].offset,
        inputs[1].offset,
        output.offset,
        start,
        end,
      );
      return;
    case "RANDN": {
      const seed = ((Date.now() & 0xffffffff) ^ ((workerIndex + 1) * 0x9e3779b1)) >>> 0;
      exports.randn(output.offset, start, end, seed);
      return;
    }
  }

  throw new Error(`WASM backend: op '${op}' not yet ported`);
}
