import type { OpParams, CoordinatorResponseData } from "../../shared/types";
import {
  TypedWorker,
  CoordinatorRequest,
  CoordinatorResponse,
  ComputeRequest,
} from "../../shared/types";

export class WorkerDispatcher {
  private coordinator: TypedWorker<CoordinatorRequest, CoordinatorResponse> | null = null;
  private sab: SharedArrayBuffer | null = null;
  private computeWorkers: Worker[] = [];
  private callbacks = new Map<string, (data: CoordinatorResponseData) => void>();
  private tensorIdCounter = 0;

  async init(threadCount = 4, memorySizeMB = 256): Promise<void> {
    if (this.coordinator) return;

    const sab = new SharedArrayBuffer(1024 * 1024 * memorySizeMB);
    this.sab = sab;

    const coordWorker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
    });
    this.coordinator = new TypedWorker(coordWorker);
    this.setupWorkerHandler(this.coordinator, "Coordinator");

    for (let i = 0; i < threadCount; i++) {
      const worker = new Worker(new URL("./worker.ts", import.meta.url), {
        type: "module",
      });
      this.computeWorkers.push(worker);

      worker.onerror = (err) => console.error(`Compute-${i} System Error:`, err);

      const channel = new MessageChannel();

      this.coordinator.postMessage(
        {
          type: "ADD_WORKER",
          payload: { workerId: i },
        },
        [channel.port1],
      );

      const initMsg: ComputeRequest = {
        type: "INIT_WORKER",
        payload: {
          workerId: i,
          role: "COMPUTE",
          buffer: sab,
        },
      };
      worker.postMessage(initMsg, [channel.port2]);
    }

    return new Promise((resolve) => {
      const reqId = this.generateId();
      this.callbacks.set(reqId, () => resolve());

      this.coordinator!.postMessage({
        type: "INIT_COORDINATOR",
        id: reqId,
        payload: {
          buffer: sab,
          totalWorkers: threadCount,
        },
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
  }

  private setupWorkerHandler(worker: TypedWorker<CoordinatorRequest, CoordinatorResponse>, name: string) {
    worker.onMessage((data) => {
      const { id, data: responseData, error } = data;

      if (id && this.callbacks.has(id)) {
        const callback = this.callbacks.get(id)!;
        if (error) {
          console.error(`${name} Error:`, error);
        } else {
          callback(responseData);
        }
        this.callbacks.delete(id);
      } else if (error) {
        console.error(`${name} Unhandled Error:`, error);
      }
    });

    worker.onError((err) => {
      console.error(`${name} System Error:`, err);
    });
  }

  nextTensorId(): string {
    return `t_${this.tensorIdCounter++}`;
  }

  allocate(tensorId: string, size: number): void {
    this.postToCoordinator({
      type: "ALLOC",
      payload: { id: tensorId, size },
    });
  }

  allocateView(tensorId: string, parentId: string, offsetBytes?: number): void {
    this.postToCoordinator({
      type: "ALLOC_VIEW",
      payload: { id: tensorId, parentId, offset: offsetBytes },
    });
  }

  free(tensorId: string): void {
    this.postToCoordinator({
      type: "FREE",
      payload: { id: tensorId },
    });
  }

  runOp(op: string, inputs: string[], output: string, params: OpParams = {}): void {
    this.postToCoordinator({
      type: "OP",
      payload: { op, inputs, output, params },
    });
  }

  set(tensorId: string, offset: number, value: number): void {
    this.postToCoordinator({
      type: "SET",
      payload: { id: tensorId, offset, value },
    });
  }

  write(tensorId: string, data: Float32Array): void {
    this.postToCoordinator({
      type: "WRITE",
      payload: { id: tensorId, data },
    });
  }

  read(tensorId: string): Promise<Float32Array> {
    return new Promise((resolve) => {
      const reqId = this.generateId();
      this.callbacks.set(reqId, (data) => {
        resolve((data as { data: Float32Array }).data);
      });

      this.postToCoordinator({
        type: "READ",
        id: reqId,
        payload: { id: tensorId },
      });
    });
  }

  /**
   * Return a zero-copy view over the SharedArrayBuffer for the given tensor id.
   * This asks the coordinator for the tensor's offset/size and creates a Float32Array
   * view on the main thread without copying the underlying data.
   * Note: caller must ensure the tensor is not being concurrently written by workers.
   */
  readView(tensorId: string): Promise<Float32Array> {
    if (!this.sab) throw new Error("Dispatcher not initialized with SharedArrayBuffer");
    return new Promise((resolve) => {
      const reqId = this.generateId();
      this.callbacks.set(reqId, (data) => {
        const { offset, size } = data as { offset: number; size: number };
        const view = new Float32Array(this.sab!, offset, size / 4);
        resolve(view);
      });

      this.postToCoordinator({
        type: "READ_VIEW",
        id: reqId,
        payload: { id: tensorId },
      });
    });
  }

  readValue(tensorId: string, offset: number): Promise<number> {
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
    return Math.random().toString(36).substring(2, 15);
  }
}
