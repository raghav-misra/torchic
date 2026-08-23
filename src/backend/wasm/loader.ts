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
    scratchPtr: number,
  ): void;

  bmm(
    aPtr: number,
    bPtr: number,
    outPtr: number,
    batchCount: number,
    m: number,
    n: number,
    k: number,
    startBatch: number,
    endBatch: number,
    scratchPtr: number,
  ): void;

  add(aPtr: number, bPtr: number, outPtr: number, start: number, end: number): void;
  sub(aPtr: number, bPtr: number, outPtr: number, start: number, end: number): void;
  mul(aPtr: number, bPtr: number, outPtr: number, start: number, end: number): void;
  div(aPtr: number, bPtr: number, outPtr: number, start: number, end: number): void;

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

  neg(aPtr: number, outPtr: number, start: number, end: number): void;
  relu(aPtr: number, outPtr: number, start: number, end: number): void;
  relu_backward(
    inputPtr: number,
    gradOutputPtr: number,
    gradInputPtr: number,
    start: number,
    end: number,
  ): void;
  exp(aPtr: number, outPtr: number, start: number, end: number): void;
  log(aPtr: number, outPtr: number, start: number, end: number): void;
  tanh(aPtr: number, outPtr: number, start: number, end: number): void;
  tanh_backward(
    outputPtr: number,
    gradOutputPtr: number,
    gradInputPtr: number,
    start: number,
    end: number,
  ): void;

  gelu(aPtr: number, outPtr: number, start: number, end: number): void;
  gelu_backward(
    inputPtr: number,
    gradOutputPtr: number,
    gradInputPtr: number,
    start: number,
    end: number,
  ): void;
  sqrt_op(aPtr: number, outPtr: number, start: number, end: number): void;
  sqrt_backward(
    outputPtr: number,
    gradOutputPtr: number,
    gradInputPtr: number,
    start: number,
    end: number,
  ): void;
  rsqrt_op(aPtr: number, outPtr: number, start: number, end: number): void;
  rsqrt_backward(
    outputPtr: number,
    gradOutputPtr: number,
    gradInputPtr: number,
    start: number,
    end: number,
  ): void;

  sigmoid(aPtr: number, outPtr: number, start: number, end: number): void;
  sigmoid_backward(
    outputPtr: number,
    gradOutputPtr: number,
    gradInputPtr: number,
    start: number,
    end: number,
  ): void;

  leaky_relu(
    aPtr: number,
    outPtr: number,
    negativeSlope: number,
    start: number,
    end: number,
  ): void;
  leaky_relu_backward(
    inputPtr: number,
    gradOutputPtr: number,
    gradInputPtr: number,
    negativeSlope: number,
    start: number,
    end: number,
  ): void;

  silu(aPtr: number, outPtr: number, start: number, end: number): void;
  silu_backward(
    inputPtr: number,
    gradOutputPtr: number,
    gradInputPtr: number,
    start: number,
    end: number,
  ): void;

  fill(outPtr: number, val: number, start: number, end: number): void;
  copy(inputPtr: number, outPtr: number, start: number, end: number): void;

  randn(outPtr: number, start: number, end: number, seed: number): void;

  sum_partial(
    inputPtr: number,
    outPtr: number,
    outIndex: number,
    start: number,
    end: number,
  ): void;
  sum_final(inputPtr: number, outPtr: number, n: number): void;
  add_scalar_tensor(
    aPtr: number,
    scalarPtr: number,
    outPtr: number,
    start: number,
    end: number,
  ): void;

  transpose(
    inputPtr: number,
    outputPtr: number,
    m: number,
    n: number,
    startRow: number,
    endRow: number,
  ): void;

  softmax2d(
    inputPtr: number,
    outputPtr: number,
    m: number,
    n: number,
    startRow: number,
    endRow: number,
  ): void;
  softmax_backward2d(
    outputPtr: number,
    gradOutputPtr: number,
    gradInputPtr: number,
    m: number,
    n: number,
    startRow: number,
    endRow: number,
  ): void;

  embedding(
    weightsPtr: number,
    indicesPtr: number,
    outputPtr: number,
    embeddingDim: number,
    start: number,
    end: number,
  ): void;
  embedding_backward(
    weightsGradPtr: number,
    indicesPtr: number,
    outputGradPtr: number,
    embeddingDim: number,
    start: number,
    end: number,
  ): void;

  materialize(
    inputPtr: number,
    outputPtr: number,
    start: number,
    end: number,
    ndim: number,
    shapePtr: number,
    stridesPtr: number,
  ): void;

  sum_axis(
    inputPtr: number,
    outputPtr: number,
    axisSize: number,
    innerSize: number,
    start: number,
    end: number,
  ): void;

  conv1d(
    inputPtr: number,
    weightPtr: number,
    biasPtr: number,
    outPtr: number,
    hasBias: number,
    bTotal: number,
    cIn: number,
    lIn: number,
    cOut: number,
    k: number,
    lOut: number,
    stride: number,
    pad: number,
    dil: number,
    groups: number,
    startBatch: number,
    endBatch: number,
  ): void;
  conv_transpose1d(
    inputPtr: number,
    weightPtr: number,
    biasPtr: number,
    outPtr: number,
    hasBias: number,
    bTotal: number,
    cIn: number,
    lIn: number,
    cOut: number,
    k: number,
    lOut: number,
    stride: number,
    pad: number,
    dil: number,
    groups: number,
    startBatch: number,
    endBatch: number,
  ): void;

  concat_slab(
    inputPtr: number,
    outPtr: number,
    outerSize: number,
    inAxisSize: number,
    outAxisSize: number,
    axisOffset: number,
    innerSize: number,
    start: number,
    end: number,
  ): void;
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
