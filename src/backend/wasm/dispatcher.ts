import type { Dispatcher } from "../dispatcher";
import type { OpParams, TensorId } from "../../shared/types";

// Skeleton implementation. The Rust/WASM kernels and worker loader land in
// follow-up commits — every method throws until then so a misconfigured
// `init({ backend: "wasm" })` fails loudly instead of silently no-op'ing.
export class WasmDispatcher implements Dispatcher {
  private notImplemented(): never {
    throw new Error("WASM backend is not yet implemented");
  }

  init(_threadCount?: number, _memorySizeMB?: number): Promise<void> {
    return this.notImplemented();
  }

  shutdown(): void {
    this.notImplemented();
  }

  nextTensorId(): TensorId {
    return this.notImplemented();
  }

  allocate(_tensorId: TensorId, _size: number): void {
    this.notImplemented();
  }

  allocateView(_tensorId: TensorId, _parentId: TensorId, _offsetBytes?: number): void {
    this.notImplemented();
  }

  free(_tensorId: TensorId): void {
    this.notImplemented();
  }

  runOp(_op: string, _inputs: TensorId[], _output: TensorId, _params?: OpParams): void {
    this.notImplemented();
  }

  set(_tensorId: TensorId, _offset: number, _value: number): void {
    this.notImplemented();
  }

  write(_tensorId: TensorId, _data: Float32Array): void {
    this.notImplemented();
  }

  read(_tensorId: TensorId): Promise<Float32Array> {
    return this.notImplemented();
  }

  readView(_tensorId: TensorId): Promise<Float32Array> {
    return this.notImplemented();
  }

  readValue(_tensorId: TensorId, _offset: number): Promise<number> {
    return this.notImplemented();
  }
}
