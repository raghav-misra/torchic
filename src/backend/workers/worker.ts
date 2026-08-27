import { MemoryAllocator } from "../memory";
import * as elementwise from "./kernels/elementwise";
import * as matmul from "./kernels/matmul";
import * as transpose from "./kernels/transpose";
import * as reductions from "./kernels/reductions";
import * as embedding from "./kernels/embedding";
import * as conv from "./kernels/conv";
import * as concat from "./kernels/concat";
import * as lstm from "./kernels/lstm";
import { defineWorkerOnMessage } from "../../shared/utils";
import type { OpParams, BufferRegion } from "../../shared/types";
import { CoordinatorRequest, ComputeRequest, ComputeResponse, TypedPort } from "../../shared/types";

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
const tensorRegistry = new Map<string, TensorMetadata>();

const computePorts: TypedPort<ComputeRequest, ComputeResponse>[] = [];
const pendingTasks = new Map<string, { resolve: () => void; count: number }>();
let commandQueue = Promise.resolve();

let coordinatorPort: TypedPort<ComputeResponse, ComputeRequest> | null = null;

self.onmessage = defineWorkerOnMessage<CoordinatorRequest | ComputeRequest>((data, ports) => {
  const { type } = data;

  if (type === "INIT_COORDINATOR") {
    role = "COORDINATOR";
    const payload = data.payload;
    buffer = payload.buffer;
    memoryAllocator = new MemoryAllocator(buffer);
    self.postMessage({ id: data.id, data: { status: "ok" } });
    return;
  }

  if (type === "INIT_WORKER") {
    role = "COMPUTE";
    const payload = data.payload;
    buffer = payload.buffer;

    const port = ports[0];
    coordinatorPort = new TypedPort(port);
    setupComputeWorker(coordinatorPort);
    return;
  }

  if (type === "ADD_WORKER") {
    if (role !== "COORDINATOR") return;
    const port = ports[0];
    const typedPort = new TypedPort<ComputeRequest, ComputeResponse>(port);
    computePorts.push(typedPort);
    setupCoordinatorPort(typedPort);
    return;
  }

  if (role === "COORDINATOR") {
    const req = data as CoordinatorRequest;

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
        console.error("Coordinator Error:", err);
        const reqId = "id" in req ? (req as { id: string }).id : undefined;
        if (reqId) {
          self.postMessage({ id: reqId, error: err.message });
        }
      });
  }
});

function setupCoordinatorPort(port: TypedPort<ComputeRequest, ComputeResponse>) {
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
    tensorRegistry.set(payload.id, {
      offset,
      size: payload.size,
      isView: false,
    });
  } catch (e: unknown) {
    console.error("Allocation failed:", e instanceof Error ? e.message : e);
  }
}

function handleAllocView(payload: { id: string; parentId: string; offset?: number }) {
  const parentMeta = tensorRegistry.get(payload.parentId);
  if (parentMeta) {
    const providedOffset = payload.offset;
    const offset =
      typeof providedOffset === "number" ? parentMeta.offset + providedOffset : parentMeta.offset;
    tensorRegistry.set(payload.id, {
      offset,
      size: parentMeta.size,
      isView: true,
    });
  } else {
    console.error(`Cannot create view: parent tensor ${payload.parentId} not found`);
  }
}

function handleFree(payload: { id: string }) {
  if (!memoryAllocator) return;
  const meta = tensorRegistry.get(payload.id);
  if (meta) {
    if (!meta.isView) {
      memoryAllocator.free(meta.offset, meta.size);
    }
    tensorRegistry.delete(payload.id);
  }
}

function handleSet(payload: { id: string; offset: number; value: number }) {
  if (!buffer) return;
  const meta = tensorRegistry.get(payload.id);
  if (meta) {
    const view = new Float32Array(buffer, meta.offset, meta.size / 4);
    view[payload.offset] = payload.value;
  }
}

function handleWrite(payload: { id: string; data: Float32Array }) {
  if (!buffer) return;
  const meta = tensorRegistry.get(payload.id);
  if (meta) {
    const view = new Float32Array(buffer, meta.offset, meta.size / 4);
    view.set(payload.data);
  }
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

  // SUM: two-phase reduce (partial sums -> final sum)
  if (payload.op === "SUM") {
    if (!memoryAllocator) return;

    const tempSize = numWorkers * 4;
    const tempOffset = memoryAllocator.allocate(tempSize);

    const taskId1 = crypto.randomUUID();
    const donePromise1 = new Promise<void>((resolve) => {
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

    await donePromise1;

    const taskId2 = crypto.randomUUID();
    const donePromise2 = new Promise<void>((resolve) => {
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

    await donePromise2;

    memoryAllocator.free(tempOffset, tempSize);

    if (reqId) {
      self.postMessage({ id: reqId, data: { status: "done" } });
    }
    return;
  }

  // Single worker to avoid race conditions on scatter-add
  if (payload.op === "EMBEDDING_BACKWARD") {
    const taskId = crypto.randomUUID();
    const donePromise = new Promise<void>((resolve) => {
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

    await donePromise;
    if (reqId) self.postMessage({ id: reqId, data: { status: "done" } });
    return;
  }

  const taskId = crypto.randomUUID();

  const donePromise = new Promise<void>((resolve) => {
    pendingTasks.set(taskId, { resolve, count: numWorkers });
  });

  computePorts.forEach((port, index) => {
    port.postMessage({
      type: "EXECUTE_TASK",
      taskId,
      op: payload.op,
      inputs: inputs.map((m) => ({ offset: m.offset, size: m.size })),
      output: { offset: output.offset, size: output.size },
      params: payload.params,
      workerIndex: index,
      totalWorkers: numWorkers,
    });
  });

  await donePromise;

  if (reqId) {
    self.postMessage({ id: reqId, data: { status: "done" } });
  }
}

function handleRead(payload: { id: string }, reqId: string) {
  const meta = tensorRegistry.get(payload.id);
  if (!meta || !buffer) return;

  const src = new Float32Array(buffer, meta.offset, meta.size / 4);
  const copy = new Float32Array(src);

  self.postMessage(
    {
      id: reqId,
      data: { data: copy },
    },
    [copy.buffer],
  );
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
  const value = view[payload.offset];

  self.postMessage({
    id: reqId,
    data: { value },
  });
}

function setupComputeWorker(port: TypedPort<ComputeResponse, ComputeRequest>) {
  port.onMessage((data) => {
    if (data.type === "EXECUTE_TASK") {
      executeKernel(
        data.op,
        data.inputs,
        data.output,
        data.params,
        data.workerIndex,
        data.totalWorkers,
      );
      port.postMessage({ type: "TASK_DONE", taskId: data.taskId });
    }
  });
}

function executeKernel(
  op: string,
  inputs: BufferRegion[],
  output: BufferRegion,
  params: OpParams,
  workerIndex: number,
  totalWorkers: number,
) {
  if (!buffer) return;
  const buf = buffer;

  const inputViews = inputs.map((meta) => new Float32Array(buf, meta.offset, meta.size / 4));
  const outputView = new Float32Array(buf, output.offset, output.size / 4);

  if (op === "MATMUL") {
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const k = required(params.k, "k");
    const rowsPerWorker = Math.ceil(m / totalWorkers);
    const startRow = workerIndex * rowsPerWorker;
    const endRow = Math.min(startRow + rowsPerWorker, m);

    if (startRow < m) {
      matmul.matmul(
        inputViews[0],
        inputViews[1],
        outputView,
        m,
        n,
        k,
        startRow,
        endRow,
        params.stridesA,
        params.stridesB,
      );
    }
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
    if (startBatch < batchCount) {
      matmul.bmm(inputViews[0], inputViews[1], outputView, batchCount, m, n, k, startBatch, endBatch);
    }
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
    const bias = params.hasBias ? inputViews[2] : null;
    const perWorker = Math.ceil(B / totalWorkers);
    const startBatch = workerIndex * perWorker;
    const endBatch = Math.min(startBatch + perWorker, B);
    if (startBatch < B) {
      const fn = op === "CONV1D" ? conv.conv1d : conv.conv_transpose1d;
      fn(
        inputViews[0],
        inputViews[1],
        bias,
        outputView,
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
    }
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
    if (start < total) {
      concat.concat_slab(
        inputViews[0],
        outputView,
        outerSize,
        inAxisSize,
        outAxisSize,
        axisOffset,
        innerSize,
        start,
        end,
      );
    }
    return;
  }

  if (op === "LSTM_STEP") {
    // Recurrence per timestep is tiny (B*H ≈ 256 cells); running on one
    // worker beats coordinating across all of them.
    if (workerIndex === 0) {
      const hidden = required(params.hidden, "hidden");
      const inSize = required(params.inSize, "inSize");
      const batchSize = required(params.batchSize, "batchSize");
      // Direct-write vs packed: if an 8th input is present, it's cOut; use
      // relative element offsets from params. Otherwise fall back to packed
      // [B, 2H] with h at output.offset, c at output.offset + B*H*4.
      let hNewOffBytes: number;
      let cNewOffBytes: number;
      if (inputs.length > 7) {
        const cOut = inputs[7];
        hNewOffBytes = output.offset + (params.hNewOffElements ?? 0) * 4;
        cNewOffBytes = cOut.offset + (params.cNewOffElements ?? 0) * 4;
      } else {
        hNewOffBytes = output.offset;
        cNewOffBytes = output.offset + batchSize * hidden * 4;
      }
      const hOutView = new Float32Array(buf, hNewOffBytes, batchSize * hidden);
      const cOutView = new Float32Array(buf, cNewOffBytes, batchSize * hidden);
      lstm.lstm_step(
        inputViews[0], inputViews[1], inputViews[2],
        inputViews[3], inputViews[4], inputViews[5], inputViews[6],
        hOutView, 0, cOutView, 0,
        batchSize, hidden, inSize,
      );
    }
    return;
  }

  if (op === "SOFTMAX") {
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const rowsPerWorker = Math.ceil(m / totalWorkers);
    const startRow = workerIndex * rowsPerWorker;
    const endRow = Math.min(startRow + rowsPerWorker, m);

    if (startRow < m) {
      elementwise.softmax2d(inputViews[0], outputView, m, n, startRow, endRow);
    }
    return;
  }

  if (op === "SOFTMAX_BACKWARD") {
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const rowsPerWorker = Math.ceil(m / totalWorkers);
    const startRow = workerIndex * rowsPerWorker;
    const endRow = Math.min(startRow + rowsPerWorker, m);

    if (startRow < m) {
      elementwise.softmax_backward2d(
        inputViews[0],
        inputViews[1],
        outputView,
        m,
        n,
        startRow,
        endRow,
      );
    }
    return;
  }

  if (op === "RMS_NORM") {
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const eps = required(params.eps, "eps");
    const rowsPerWorker = Math.ceil(m / totalWorkers);
    const startRow = workerIndex * rowsPerWorker;
    const endRow = Math.min(startRow + rowsPerWorker, m);

    if (startRow < m) {
      elementwise.rms_norm2d(
        inputViews[0],
        inputViews[1],
        outputView,
        m,
        n,
        eps,
        startRow,
        endRow,
      );
    }
    return;
  }

  if (op === "ROPE") {
    const m = required(params.m, "m");
    const tSeq = required(params.tSeq, "tSeq");
    const dHalf = required(params.dHalf, "dHalf");
    const rowsPerWorker = Math.ceil(m / totalWorkers);
    const startRow = workerIndex * rowsPerWorker;
    const endRow = Math.min(startRow + rowsPerWorker, m);

    if (startRow < m) {
      elementwise.rope(
        inputViews[0],
        inputViews[1],
        inputViews[2],
        outputView,
        tSeq,
        dHalf,
        startRow,
        endRow,
      );
    }
    return;
  }

  if (op === "CAUSAL_SOFTMAX") {
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const pastLen = params.pastLen ?? 0;
    const tQuery = params.tQuery ?? m;
    const rowsPerWorker = Math.ceil(m / totalWorkers);
    const startRow = workerIndex * rowsPerWorker;
    const endRow = Math.min(startRow + rowsPerWorker, m);

    if (startRow < m) {
      elementwise.causal_softmax2d(
        inputViews[0],
        outputView,
        m,
        n,
        pastLen,
        tQuery,
        startRow,
        endRow,
      );
    }
    return;
  }

  if (op === "COPY_RANGE") {
    const count = required(params.count, "count");
    const dstOffset = required(params.dstOffset, "dstOffset");
    const chunk = Math.ceil(count / totalWorkers);
    const start = workerIndex * chunk;
    const end = Math.min(start + chunk, count);

    if (start < count) {
      elementwise.copy_range(inputViews[0], outputView, dstOffset, start, end);
    }
    return;
  }

  if (op === "REPEAT_INTERLEAVE") {
    const count = required(params.count, "count");
    const axisSize = required(params.axisSize, "axisSize");
    const innerSize = required(params.innerSize, "innerSize");
    const repeats = required(params.repeats, "repeats");
    const chunk = Math.ceil(count / totalWorkers);
    const start = workerIndex * chunk;
    const end = Math.min(start + chunk, count);

    if (start < count) {
      elementwise.repeat_interleave(
        inputViews[0],
        outputView,
        axisSize,
        innerSize,
        repeats,
        start,
        end,
      );
    }
    return;
  }

  if (op === "TRANSPOSE") {
    const m = required(params.m, "m");
    const n = required(params.n, "n");
    const rowsPerWorker = Math.ceil(n / totalWorkers);
    const startRow = workerIndex * rowsPerWorker;
    const endRow = Math.min(startRow + rowsPerWorker, n);

    if (startRow < n) {
      transpose.transpose(inputViews[0], outputView, m, n, startRow, endRow);
    }
    return;
  }

  if (op === "SUM_PARTIAL") {
    const outIndex = required(params.outIndex, "outIndex");
    const totalElements = inputViews[0].length;
    const chunkSize = Math.ceil(totalElements / totalWorkers);
    const start = workerIndex * chunkSize;
    const end = Math.min(start + chunkSize, totalElements);

    if (start < totalElements) {
      reductions.sum_partial(inputViews[0], outputView, outIndex, start, end);
    } else {
      outputView[outIndex] = 0;
    }
    return;
  }

  if (op === "SUM_FINAL") {
    reductions.sum_final(inputViews[0], outputView, required(params.n, "n"));
    return;
  }

  const totalElements = outputView.length;
  const chunkSize = Math.ceil(totalElements / totalWorkers);
  const start = workerIndex * chunkSize;
  const end = Math.min(start + chunkSize, totalElements);

  if (start >= totalElements) return;

  switch (op) {
    case "ADD":
      elementwise.add(
        inputViews[0],
        inputViews[1],
        outputView,
        start,
        end,
        params.shape,
        params.stridesA,
        params.stridesB,
      );
      break;
    case "SUB":
      elementwise.sub(
        inputViews[0],
        inputViews[1],
        outputView,
        start,
        end,
        params.shape,
        params.stridesA,
        params.stridesB,
      );
      break;
    case "MUL":
      elementwise.mul(
        inputViews[0],
        inputViews[1],
        outputView,
        start,
        end,
        params.shape,
        params.stridesA,
        params.stridesB,
      );
      break;
    case "DIV":
      elementwise.div(
        inputViews[0],
        inputViews[1],
        outputView,
        start,
        end,
        params.shape,
        params.stridesA,
        params.stridesB,
      );
      break;
    case "RELU":
      elementwise.relu(inputViews[0], outputView, start, end, params.shape, params.strides);
      break;
    case "RELU_BACKWARD":
      elementwise.relu_backward(
        inputViews[0],
        inputViews[1],
        outputView,
        start,
        end,
        params.shape,
        params.strides,
      );
      break;
    case "EXP":
      elementwise.exp(inputViews[0], outputView, start, end, params.shape, params.strides);
      break;
    case "TANH":
      elementwise.tanh(inputViews[0], outputView, start, end, params.shape, params.strides);
      break;
    case "TANH_BACKWARD":
      elementwise.tanh_backward(inputViews[0], inputViews[1], outputView, start, end);
      break;
    case "SIN":
      elementwise.sin(inputViews[0], outputView, start, end, params.shape, params.strides);
      break;
    case "COS":
      elementwise.cos(inputViews[0], outputView, start, end, params.shape, params.strides);
      break;
    case "GELU":
      elementwise.gelu(inputViews[0], outputView, start, end, params.shape, params.strides);
      break;
    case "GELU_BACKWARD":
      elementwise.gelu_backward(
        inputViews[0],
        inputViews[1],
        outputView,
        start,
        end,
        params.shape,
        params.strides,
      );
      break;
    case "SQRT":
      elementwise.sqrt(inputViews[0], outputView, start, end, params.shape, params.strides);
      break;
    case "SQRT_BACKWARD":
      elementwise.sqrt_backward(inputViews[0], inputViews[1], outputView, start, end);
      break;
    case "RSQRT":
      elementwise.rsqrt(inputViews[0], outputView, start, end, params.shape, params.strides);
      break;
    case "RSQRT_BACKWARD":
      elementwise.rsqrt_backward(inputViews[0], inputViews[1], outputView, start, end);
      break;
    case "SIGMOID":
      elementwise.sigmoid(inputViews[0], outputView, start, end, params.shape, params.strides);
      break;
    case "SIGMOID_BACKWARD":
      elementwise.sigmoid_backward(inputViews[0], inputViews[1], outputView, start, end);
      break;
    case "LEAKY_RELU":
      elementwise.leaky_relu(
        inputViews[0],
        outputView,
        required(params.negativeSlope, "negativeSlope"),
        start,
        end,
        params.shape,
        params.strides,
      );
      break;
    case "LEAKY_RELU_BACKWARD":
      elementwise.leaky_relu_backward(
        inputViews[0],
        inputViews[1],
        outputView,
        required(params.negativeSlope, "negativeSlope"),
        start,
        end,
      );
      break;
    case "SILU":
      elementwise.silu(inputViews[0], outputView, start, end, params.shape, params.strides);
      break;
    case "SILU_BACKWARD":
      elementwise.silu_backward(inputViews[0], inputViews[1], outputView, start, end);
      break;
    case "LOG":
      elementwise.log(inputViews[0], outputView, start, end, params.shape, params.strides);
      break;
    case "NEG":
      elementwise.neg(inputViews[0], outputView, start, end, params.shape, params.strides);
      break;
    case "FILL":
      elementwise.fill(outputView, required(params.value, "value"), start, end);
      break;
    case "RANDN":
      elementwise.randn(outputView, start, end);
      break;
    case "SUM_AXIS":
      reductions.sum_axis(
        inputViews[0],
        outputView,
        start,
        end,
        required(params.shape, "shape"),
        required(params.strides, "strides"),
        required(params.axis, "axis"),
      );
      break;
    case "ADD_SCALAR_TENSOR":
      reductions.add_scalar_tensor(inputViews[0], inputViews[1], outputView, start, end);
      break;
    case "COPY":
      elementwise.copy(inputViews[0], outputView, start, end);
      break;
    case "MATERIALIZE":
      elementwise.materialize(
        inputViews[0],
        outputView,
        start,
        end,
        required(params.shape, "shape"),
        required(params.strides, "strides"),
      );
      break;
    case "EMBEDDING":
      embedding.embedding(
        inputViews[0],
        inputViews[1],
        outputView,
        required(params.embeddingDim, "embeddingDim"),
        start,
        end,
      );
      break;
    case "EMBEDDING_BACKWARD":
      embedding.embedding_backward(
        outputView,
        inputViews[0],
        inputViews[1],
        required(params.embeddingDim, "embeddingDim"),
        start,
        end,
      );
      break;
    default:
      console.error(`Unknown op: ${op}`);
  }
}
