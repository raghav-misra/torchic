import { WorkerDispatcher } from "../backend/workers/dispatcher";
import { WasmDispatcher } from "../backend/wasm/dispatcher";
import type { Dispatcher } from "../backend/dispatcher";

let dispatcher: Dispatcher | null = null;

export function getDispatcher(): Dispatcher {
  if (!dispatcher) {
    throw new Error("Torchic not initialized. Call init() before using tensors.");
  }
  return dispatcher;
}

export function isDispatcherReady(): boolean {
  return dispatcher !== null;
}

interface InitOptions {
  backend: "workers" | "wasm";
  threadCount?: number;
  memorySizeMB?: number;
}

export async function init(options: InitOptions) {
  if (options.backend === "workers") {
    dispatcher = new WorkerDispatcher();
    await dispatcher.init(options.threadCount, options.memorySizeMB);
    return;
  }
  if (options.backend === "wasm") {
    dispatcher = new WasmDispatcher();
    await dispatcher.init(options.threadCount, options.memorySizeMB);
    return;
  }
}

export function shutdown() {
  if (!dispatcher) return;
  dispatcher.shutdown();
  dispatcher = null;
}
