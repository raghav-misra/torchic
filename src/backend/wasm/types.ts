import type {
  CoordinatorRequest,
  CoordinatorResponse,
  ComputeRequest,
  ComputeResponse,
} from "../../shared/types";

export type WasmInitCoordinator = {
  type: "INIT_WASM_COORDINATOR";
  id: string;
  payload: { memory: WebAssembly.Memory; totalWorkers: number };
};

export type WasmInitWorker = {
  type: "INIT_WASM_WORKER";
  payload: {
    workerId: number;
    module: WebAssembly.Module;
    memory: WebAssembly.Memory;
  };
};

export type WasmCoordinatorRequest = CoordinatorRequest | WasmInitCoordinator;
export type WasmComputeRequest = ComputeRequest | WasmInitWorker;

export type { CoordinatorResponse, ComputeResponse };
