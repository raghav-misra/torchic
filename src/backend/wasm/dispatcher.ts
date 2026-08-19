import type { OpParams, CoordinatorResponseData, TensorId } from "../../shared/types";
import { TypedWorker, CoordinatorRequest, CoordinatorResponse } from "../../shared/types";
import type { Dispatcher } from "../dispatcher";
import type { WasmCoordinatorRequest, WasmComputeRequest } from "./types";

export class WasmDispatcher implements Dispatcher {
  private coordinator: TypedWorker<WasmCoordinatorRequest, CoordinatorResponse> | null = null;
  private sab: SharedArrayBuffer | null = null;
  private computeWorkers: Worker[] = [];
  private callbacks = new Map<string, (data: CoordinatorResponseData) => void>();
  private tensorIdCounter = 0;
  private readonly instanceTag = crypto.randomUUID().slice(0, 8);

  async init(threadCount = 4, memorySizeMB = 256): Promise<void> {
    if (this.coordinator) return;

    // Lazy so ?url resolution of kernels.wasm only happens when the WASM backend
    // is actually selected; otherwise Vitest / dev builds without the artifact fail.
    const { compileKernels, createSharedMemory } = await import("./loader");

    const [module, memory] = await Promise.all([
      compileKernels(),
      Promise.resolve(createSharedMemory(memorySizeMB)),
    ]);
    this.sab = memory.buffer as unknown as SharedArrayBuffer;

    const coordWorker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.coordinator = new TypedWorker(coordWorker);
    this.setupWorkerHandler(this.coordinator, "WasmCoordinator");

    for (let i = 0; i < threadCount; i++) {
      const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
      this.computeWorkers.push(worker);
      worker.onerror = (err) => console.error(`WasmCompute-${i} System Error:`, err);

      const channel = new MessageChannel();
      this.coordinator.postMessage({ type: "ADD_WORKER", payload: { workerId: i } }, [
        channel.port1,
      ]);

      const initMsg: WasmComputeRequest = {
        type: "INIT_WASM_WORKER",
        payload: { workerId: i, module, memory },
      };
      worker.postMessage(initMsg, [channel.port2]);
    }

    const coordinator = this.coordinator;
    return new Promise((resolve) => {
      const reqId = this.generateId();
      this.callbacks.set(reqId, () => resolve());

      coordinator.postMessage({
        type: "INIT_WASM_COORDINATOR",
        id: reqId,
        payload: { memory, totalWorkers: threadCount },
      });
    });
  }

  shutdown(): void {
    try {
      if (this.coordinator) {
        this.coordinator.terminate();
        this.coordinator = null;
      }
    } catch (e) {
      console.warn("Error terminating coordinator:", e);
    }

    try {
      for (const w of this.computeWorkers) {
        try {
          w.terminate();
        } catch (e) {
          console.warn("Error terminating compute worker:", e);
        }
      }
    } finally {
      this.computeWorkers = [];
    }

    this.callbacks.clear();
    this.tensorIdCounter = 0;
    this.sab = null;
  }

  private setupWorkerHandler(
    worker: TypedWorker<WasmCoordinatorRequest, CoordinatorResponse>,
    name: string,
  ) {
    worker.onMessage((data) => {
      const { id, data: responseData, error } = data;
      if (id && this.callbacks.has(id)) {
        const cb = this.callbacks.get(id);
        if (!cb) return;
        if (error) console.error(`${name} Error:`, error);
        else cb(responseData);
        this.callbacks.delete(id);
      } else if (error) {
        console.error(`${name} Unhandled Error:`, error);
      }
    });
    worker.onError((err) => console.error(`${name} System Error:`, err));
  }

  nextTensorId(): TensorId {
    return `${this.instanceTag}_t_${this.tensorIdCounter++}`;
  }

  allocate(tensorId: TensorId, size: number): void {
    this.postToCoordinator({ type: "ALLOC", payload: { id: tensorId, size } });
  }

  allocateView(tensorId: TensorId, parentId: TensorId, offsetBytes?: number): void {
    this.postToCoordinator({
      type: "ALLOC_VIEW",
      payload: { id: tensorId, parentId, offset: offsetBytes },
    });
  }

  free(tensorId: TensorId): void {
    this.postToCoordinator({ type: "FREE", payload: { id: tensorId } });
  }

  runOp(op: string, inputs: TensorId[], output: TensorId, params: OpParams = {}): void {
    this.postToCoordinator({ type: "OP", payload: { op, inputs, output, params } });
  }

  set(tensorId: TensorId, offset: number, value: number): void {
    this.postToCoordinator({ type: "SET", payload: { id: tensorId, offset, value } });
  }

  write(tensorId: TensorId, data: Float32Array): void {
    this.postToCoordinator({ type: "WRITE", payload: { id: tensorId, data } });
  }

  read(tensorId: TensorId): Promise<Float32Array> {
    return new Promise((resolve) => {
      const reqId = this.generateId();
      this.callbacks.set(reqId, (data) => {
        resolve((data as { data: Float32Array }).data);
      });
      this.postToCoordinator({ type: "READ", id: reqId, payload: { id: tensorId } });
    });
  }

  readView(tensorId: TensorId): Promise<Float32Array> {
    const sab = this.sab;
    if (!sab) throw new Error("Dispatcher not initialized");
    return new Promise((resolve) => {
      const reqId = this.generateId();
      this.callbacks.set(reqId, (data) => {
        const { offset, size } = data as { offset: number; size: number };
        resolve(new Float32Array(sab, offset, size / 4));
      });
      this.postToCoordinator({ type: "READ_VIEW", id: reqId, payload: { id: tensorId } });
    });
  }

  readValue(tensorId: TensorId, offset: number): Promise<number> {
    return new Promise((resolve) => {
      const reqId = this.generateId();
      this.callbacks.set(reqId, (data) => {
        resolve((data as { value: number }).value);
      });
      this.postToCoordinator({
        type: "READ_VALUE",
        id: reqId,
        payload: { id: tensorId, offset },
      });
    });
  }

  private postToCoordinator(command: CoordinatorRequest) {
    if (!this.coordinator) {
      throw new Error("Dispatcher not initialized. Call init() first.");
    }
    this.coordinator.postMessage(command);
  }

  private generateId(): string {
    return crypto.randomUUID();
  }
}
