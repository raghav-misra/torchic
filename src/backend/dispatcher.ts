import type { OpParams, TensorId } from "../shared/types";

export interface Dispatcher {
  init(threadCount?: number, memorySizeMB?: number): Promise<void>;
  shutdown(): void;

  nextTensorId(): TensorId;
  allocate(tensorId: TensorId, size: number): void;
  allocateView(tensorId: TensorId, parentId: TensorId, offsetBytes?: number): void;
  free(tensorId: TensorId): void;

  runOp(op: string, inputs: TensorId[], output: TensorId, params?: OpParams): void;

  set(tensorId: TensorId, offset: number, value: number): void;
  write(tensorId: TensorId, data: Float32Array): void;

  read(tensorId: TensorId): Promise<Float32Array>;
  readView(tensorId: TensorId): Promise<Float32Array>;
  readValue(tensorId: TensorId, offset: number): Promise<number>;
}
