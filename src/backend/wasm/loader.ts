import kernelsUrl from "./kernels.wasm?url";

const PAGES_PER_MB = 16; // 64 KiB pages
const MAX_MEMORY_MB = 1024; // matches --max-memory=1073741824 in the Rust build

export interface KernelExports {
  matmul(
    aPtr: number,
    bPtr: number,
    outPtr: number,
    m: number,
    n: number,
    k: number,
    startRow: number,
    endRow: number,
    aRowStride: number,
    aColStride: number,
    bRowStride: number,
    bColStride: number,
  ): void;

  add(aPtr: number, bPtr: number, outPtr: number, start: number, end: number): void;

  add_broadcast(
    aPtr: number,
    bPtr: number,
    outPtr: number,
    start: number,
    end: number,
    ndim: number,
    shapePtr: number,
    stridesAPtr: number,
    stridesBPtr: number,
  ): void;

  randn(outPtr: number, start: number, end: number, seed: number): void;
}

export interface WasmInstance {
  memory: WebAssembly.Memory;
  exports: KernelExports;
}

export async function compileKernels(): Promise<WebAssembly.Module> {
  const res = await fetch(kernelsUrl);
  return WebAssembly.compileStreaming(res);
}

export function createSharedMemory(memorySizeMB: number): WebAssembly.Memory {
  if (memorySizeMB > MAX_MEMORY_MB) {
    throw new Error(
      `memorySizeMB=${memorySizeMB} exceeds WASM module's compiled max of ${MAX_MEMORY_MB}MB`,
    );
  }
  const initial = memorySizeMB * PAGES_PER_MB;
  const maximum = MAX_MEMORY_MB * PAGES_PER_MB;
  return new WebAssembly.Memory({ initial, maximum, shared: true });
}

export async function instantiateKernels(
  module: WebAssembly.Module,
  memory: WebAssembly.Memory,
): Promise<WasmInstance> {
  const instance = await WebAssembly.instantiate(module, { env: { memory } });
  return {
    memory,
    exports: instance.exports as unknown as KernelExports,
  };
}
