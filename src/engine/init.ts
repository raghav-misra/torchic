import { WorkerDispatcher } from "../backend/workers/dispatcher";

export let dispatcherInstance: WorkerDispatcher | null;

type InitOptions = {
  backend: "workers";
  threadCount?: number;
  memorySizeMB?: number;
};

export async function init(options: InitOptions) {
  if (options.backend === "workers") {
    dispatcherInstance = new WorkerDispatcher();
    await dispatcherInstance.init(options.threadCount, options.memorySizeMB);
  }
}

export async function shutdown() {
  dispatcherInstance!.shutdown();
  dispatcherInstance = null;
}
