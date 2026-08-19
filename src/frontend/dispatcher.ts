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
    const d = new WorkerDispatcher();
    await d.init(options.threadCount, options.memorySizeMB);
    dispatcher = d;
    return;
  }
  if (options.backend === "wasm") {
    const d = new WasmDispatcher();
    await d.init(options.threadCount, options.memorySizeMB);
    dispatcher = d;
    return;
  }
}

export function shutdown() {
  if (!dispatcher) return;
  dispatcher.shutdown();
  dispatcher = null;
}
