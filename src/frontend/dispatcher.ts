import { WorkerDispatcher } from "../backend/workers/dispatcher";

let dispatcher: WorkerDispatcher | null = null;

export function getDispatcher(): WorkerDispatcher {
  if (!dispatcher) {
    throw new Error("Torchic not initialized. Call init() before using tensors.");
  }
  return dispatcher;
}

export function isDispatcherReady(): boolean {
  return dispatcher !== null;
}

interface InitOptions {
  backend: "workers";
  threadCount?: number;
  memorySizeMB?: number;
}

export async function init(options: InitOptions) {
  if (options.backend === "workers") {
    dispatcher = new WorkerDispatcher();
    await dispatcher.init(options.threadCount, options.memorySizeMB);
  }
}

export function shutdown() {
  if (!dispatcher) return;
  dispatcher.shutdown();
  dispatcher = null;
}
