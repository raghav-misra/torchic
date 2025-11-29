import { WorkerDispatcher } from "../backend/workers/dispatcher";

export let dispatcher: WorkerDispatcher | null;

type InitOptions = {
  backend: "workers";
  threadCount?: number;
  memorySizeMB?: number;
};

export async function init(options: InitOptions) {
  if (options.backend === "workers") {
    dispatcher = new WorkerDispatcher();
    await dispatcher.init(options.threadCount, options.memorySizeMB);
  }
}

export function shutdown() {
  dispatcher!.shutdown();
  dispatcher = null;
}
