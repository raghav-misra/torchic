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
      memoryAllocator = new MemoryAllocator(buffer);
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

  // SUM would require SUM_PARTIAL + SUM_FINAL exports (not ported yet).
  // Let it hit the generic dispatch below so the compute worker throws a clear error.

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
  if (reqId) self.postMessage({ id: reqId, data: { status: "done" } });
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
    );
    return;
  }

  if (op === "ADD") {
    const totalElements = output.size / 4;
    const chunkSize = Math.ceil(totalElements / totalWorkers);
    const start = workerIndex * chunkSize;
    const end = Math.min(start + chunkSize, totalElements);
    if (start >= totalElements) return;

    const contiguous =
      isContiguous(params.shape, params.stridesA) &&
      isContiguous(params.shape, params.stridesB);
    if (!contiguous) {
      throw new Error("WASM backend: broadcasted ADD not yet ported");
    }
    exports.add(inputs[0].offset, inputs[1].offset, output.offset, start, end);
    return;
  }

  if (op === "RANDN") {
    const totalElements = output.size / 4;
    const chunkSize = Math.ceil(totalElements / totalWorkers);
    const start = workerIndex * chunkSize;
    const end = Math.min(start + chunkSize, totalElements);
    if (start >= totalElements) return;

    // Mix workerIndex + wall clock so parallel streams stay independent across calls.
    const seed = ((Date.now() & 0xffffffff) ^ ((workerIndex + 1) * 0x9e3779b1)) >>> 0;
    exports.randn(output.offset, start, end, seed);
    return;
  }

  throw new Error(`WASM backend: op '${op}' not yet ported`);
}
