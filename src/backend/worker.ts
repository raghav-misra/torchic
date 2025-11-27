import { MemoryAllocator } from "./memory";
import * as elementwise from "../kernels/elementwise";
import * as matmul from "../kernels/matmul";
import * as transpose from "../kernels/transpose";
import * as reductions from "../kernels/reductions";
import { defineWorkerOnMessage } from "../utils";
import {
  CoordinatorRequest,
  ComputeRequest,
  ComputeResponse,
  TypedPort,
} from "../types";

// Types
type WorkerRole = "COORDINATOR" | "COMPUTE";

interface TensorMetadata {
  offset: number;
  size: number;
}

// Global State
let role: WorkerRole = "COORDINATOR"; // Default, changes on init
let memoryAllocator: MemoryAllocator | null = null;
let buffer: SharedArrayBuffer | null = null;
let tensorRegistry: Map<string, TensorMetadata> = new Map();

// Coordinator State
let computePorts: TypedPort<ComputeRequest, ComputeResponse>[] = [];
let pendingTasks: Map<string, { resolve: () => void; count: number }> =
  new Map();
let commandQueue = Promise.resolve(); // Serialization queue for Coordinator

// Compute State
let coordinatorPort: TypedPort<ComputeResponse, ComputeRequest> | null = null;

// --- Message Handlers ---

// We use a union type for the handler because the worker can receive both types of messages
// depending on its role (initially it receives CoordinatorRequest or ComputeRequest)
// Actually, the main thread sends CoordinatorRequest (mostly) or ComputeRequest (INIT_WORKER)
// Let's just cast inside for simplicity or use a union type if we want to be strict.
// Since defineWorkerOnMessage takes a generic T, we can use CoordinatorRequest | ComputeRequest

self.onmessage = defineWorkerOnMessage<CoordinatorRequest | ComputeRequest>(
  (data, ports) => {
    const { type } = data;

    if (type === "INIT_COORDINATOR") {
      role = "COORDINATOR";
      const payload = (data as any).payload; // TS might struggle with the union discrimination here without a switch
      buffer = payload.buffer;
      memoryAllocator = new MemoryAllocator(buffer!);
      // Acknowledge init
      self.postMessage({ id: (data as any).id, data: { status: "ok" } });
      return;
    }

    if (type === "INIT_WORKER") {
      role = "COMPUTE";
      const payload = (data as any).payload;
      buffer = payload.buffer;

      // The port comes in the transfer list
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

    // Coordinator Commands
    if (role === "COORDINATOR") {
      const req = data as CoordinatorRequest;
      // We must serialize commands that touch memory or depend on previous ops
      // ALLOC/FREE are sync and fast, but to be safe and simple, let's queue everything
      // or at least queue OP and READ.

      commandQueue = commandQueue
        .then(async () => {
          switch (req.type) {
            case "ALLOC":
              handleAlloc(req.payload);
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
              await handleOp(
                { ...req.payload, params: req.payload.params || {} },
                req.id
              );
              break;
            case "READ":
              handleRead(req.payload, req.id);
              break;
            case "READ_VALUE":
              handleReadValue(req.payload, req.id);
              break;
          }
        })
        .catch((err) => {
          console.error("Coordinator Error:", err);
          // If we have an ID, we should probably reply with error, but for now just log
          const reqId = (req as any).id;
          if (reqId) {
            self.postMessage({ id: reqId, error: err.message });
          }
        });
    }
  }
);

// --- Coordinator Logic ---

function setupCoordinatorPort(
  port: TypedPort<ComputeRequest, ComputeResponse>
) {
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
    tensorRegistry.set(payload.id, { offset, size: payload.size });
  } catch (e: any) {
    console.error("Allocation failed:", e.message);
  }
}

function handleFree(payload: { id: string }) {
  if (!memoryAllocator) return;
  const meta = tensorRegistry.get(payload.id);
  if (meta) {
    memoryAllocator.free(meta.offset, meta.size);
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
  payload: { op: string; inputs: string[]; output: string; params: any },
  reqId?: string
) {
  // 1. Resolve Tensor IDs to Offsets
  const inputMetas = payload.inputs.map((id) => tensorRegistry.get(id));
  const outputMeta = tensorRegistry.get(payload.output);

  if (inputMetas.some((m) => !m) || !outputMeta) {
    console.error("Missing tensor metadata for op:", payload.op);
    return;
  }

  const numWorkers = computePorts.length;

  // Special Handling for SUM (Coordinator Logic)
  if (payload.op === 'SUM') {
      if (!memoryAllocator) return;
      
      // 1. Allocate Temp Buffer for Partial Sums
      const tempSize = numWorkers * 4;
      const tempOffset = memoryAllocator.allocate(tempSize);
      
      // 2. Dispatch SUM_PARTIAL
      const taskId1 = Math.random().toString(36).substring(7);
      const donePromise1 = new Promise<void>((resolve) => {
          pendingTasks.set(taskId1, { resolve, count: numWorkers });
      });

      computePorts.forEach((port, index) => {
          port.postMessage({
              type: "EXECUTE_TASK",
              taskId: taskId1,
              op: "SUM_PARTIAL",
              inputs: [{ offset: inputMetas[0]!.offset, size: inputMetas[0]!.size }],
              output: { offset: tempOffset, size: tempSize }, // Workers write to specific index here
              params: { outIndex: index },
              workerIndex: index,
              totalWorkers: numWorkers,
          });
      });

      await donePromise1;

      // 3. Dispatch SUM_FINAL (Single Worker)
      const taskId2 = Math.random().toString(36).substring(7);
      const donePromise2 = new Promise<void>((resolve) => {
          pendingTasks.set(taskId2, { resolve, count: 1 });
      });

      computePorts[0].postMessage({
          type: "EXECUTE_TASK",
          taskId: taskId2,
          op: "SUM_FINAL",
          inputs: [{ offset: tempOffset, size: tempSize }],
          output: { offset: outputMeta!.offset, size: outputMeta!.size },
          params: { n: numWorkers },
          workerIndex: 0,
          totalWorkers: 1,
      });

      await donePromise2;

      // 4. Free Temp
      memoryAllocator.free(tempOffset, tempSize);

      if (reqId) {
          self.postMessage({ id: reqId, data: { status: "done" } });
      }
      return;
  }

  // 2. Split the work (Sharding Logic)
  const taskId = Math.random().toString(36).substring(7);

  const donePromise = new Promise<void>((resolve) => {
    pendingTasks.set(taskId, { resolve, count: numWorkers });
  });

  // 3. Dispatch to Compute Workers
  computePorts.forEach((port, index) => {
    port.postMessage({
      type: "EXECUTE_TASK",
      taskId,
      op: payload.op,
      inputs: inputMetas.map((m) => ({ offset: m!.offset, size: m!.size })),
      output: { offset: outputMeta!.offset, size: outputMeta!.size },
      params: payload.params,
      workerIndex: index,
      totalWorkers: numWorkers,
    });
  });

  // 4. Wait and Reply
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
    [copy.buffer]
  );
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

// --- Compute Logic ---

function setupComputeWorker(port: TypedPort<ComputeResponse, ComputeRequest>) {
  port.onMessage((data) => {
    if (data.type === "EXECUTE_TASK") {
      executeKernel(
        data.op,
        data.inputs,
        data.output,
        data.params,
        data.workerIndex,
        data.totalWorkers
      );
      port.postMessage({ type: "TASK_DONE", taskId: data.taskId });
    }
  });
}

function executeKernel(
  op: string,
  inputs: any[],
  output: any,
  params: any,
  workerIndex: number,
  totalWorkers: number
) {
  if (!buffer) return;

  const inputViews = inputs.map(
    (meta) => new Float32Array(buffer!, meta.offset, meta.size / 4)
  );
  const outputView = new Float32Array(buffer!, output.offset, output.size / 4);

  // Special handling for MatMul (Row-based sharding)
  if (op === "MATMUL") {
    const { m, n, k } = params;
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
        endRow
      );
    }
    return;
  }

  // Special handling for Transpose
  if (op === "TRANSPOSE") {
    const { m, n } = params; // Input shape [m, n], Output shape [n, m]
    // We shard the OUTPUT rows (0 to n)
    const rowsPerWorker = Math.ceil(n / totalWorkers);
    const startRow = workerIndex * rowsPerWorker;
    const endRow = Math.min(startRow + rowsPerWorker, n);

    if (startRow < n) {
      transpose.transpose(
        inputViews[0],
        outputView,
        m,
        n,
        startRow,
        endRow
      );
    }
    return;
  }

  if (op === "SUM_PARTIAL") {
      // Input: Large array
      // Output: Small array (size = numWorkers)
      // We write to output[params.outIndex]
      // We sum input[start...end]
      
      // Re-calculate start/end for the INPUT array
      const totalElements = inputViews[0].length;
      const chunkSize = Math.ceil(totalElements / totalWorkers);
      const start = workerIndex * chunkSize;
      const end = Math.min(start + chunkSize, totalElements);

      if (start < totalElements) {
          reductions.sum_partial(inputViews[0], outputView, params.outIndex, start, end);
      } else {
          outputView[params.outIndex] = 0;
      }
      return;
  }

  if (op === "SUM_FINAL") {
      // Input: Small array (partial sums)
      // Output: Scalar (size 1)
      reductions.sum_final(inputViews[0], outputView, params.n);
      return;
  }

  // Default: Element-wise (Flat sharding)
  const totalElements = outputView.length;
  const chunkSize = Math.ceil(totalElements / totalWorkers);
  const start = workerIndex * chunkSize;
  const end = Math.min(start + chunkSize, totalElements);

  if (start >= totalElements) return;

  switch (op) {
    case "ADD":
      elementwise.add(inputViews[0], inputViews[1], outputView, start, end, params.shape, params.stridesA, params.stridesB);
      break;
    case "SUB":
      elementwise.sub(inputViews[0], inputViews[1], outputView, start, end, params.shape, params.stridesA, params.stridesB);
      break;
    case "MUL":
      elementwise.mul(inputViews[0], inputViews[1], outputView, start, end, params.shape, params.stridesA, params.stridesB);
      break;
    case "DIV":
      elementwise.div(inputViews[0], inputViews[1], outputView, start, end, params.shape, params.stridesA, params.stridesB);
      break;
    case "RELU":
      elementwise.relu(inputViews[0], outputView, start, end);
      break;
    case "RELU_BACKWARD":
      elementwise.relu_backward(inputViews[0], inputViews[1], outputView, start, end);
      break;
    case "EXP":
      elementwise.exp(inputViews[0], outputView, start, end);
      break;
    case "LOG":
      elementwise.log(inputViews[0], outputView, start, end);
      break;
    case "FILL":
      elementwise.fill(outputView, params.value, start, end);
      break;
    case "RANDN":
      elementwise.randn(outputView, start, end);
      break;
    case "SUM_AXIS":
      reductions.sum_axis(inputViews[0], outputView, start, end, params.shape, params.strides, params.axis);
      break;
    case "ADD_SCALAR_TENSOR":
      reductions.add_scalar_tensor(inputViews[0], inputViews[1], outputView, start, end);
      break;
    case "COPY":
      elementwise.copy(inputViews[0], outputView, start, end);
      break;
    default:
      console.error(`Unknown op: ${op}`);
  }
}
