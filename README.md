# 🔥 torchic

A toy neural network library that runs in the browser across three interchangeable compute backends: JS Web Workers, SIMD Rust compiled to WebAssembly, and WebGPU.

## Overview

Write sync-looking PyTorch-style code in JavaScript. Compute runs off the main thread on Web Workers or the GPU queue. Includes reverse-mode autograd, one shared-memory tensor heap, and zero-copy views.

### Key Features

- Three backends behind one `Dispatcher` interface. 
- Reverse-mode autograd over a dynamic computation graph.
- One `SharedArrayBuffer` heap, sub-allocated by a segregated free-list with O(1) block recycling.
- Zero-copy reshape/transpose/slice views (strides only).

## Installation

```bash
npm install torchic
```

torchic is a browser library. It requires:

1. A bundler that understands Vite-style asset imports (`?url`, `?raw`) and the `new Worker(new URL(..., import.meta.url), { type: "module" })` pattern. Vite works out of the box; Webpack 5, Rollup, and Parcel work with their asset-module features enabled.
2. Cross-origin isolation for `SharedArrayBuffer`. The dev/prod server must send these two headers on every response:
   ```
   Cross-Origin-Opener-Policy: same-origin
   Cross-Origin-Embedder-Policy: require-corp
   ```
3. A recent Chromium-based browser for the WebGPU backend (Chrome/Edge 113+ on supported hardware). The `workers` and `wasm` backends run anywhere modern.

### Vite setup

`vite.config.ts`:

```ts
import { defineConfig } from "vite";

export default defineConfig({
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  worker: { format: "es" },
  optimizeDeps: { exclude: ["torchic"] },
});
```

Then in your app:

```ts
import { Tensor, init } from "torchic";

await init({ backend: "wasm", threadCount: 4 });

const a = Tensor.fromData([1, 2, 3, 4, 5, 6], [2, 3]);
const b = Tensor.fromData([7, 8, 9, 10, 11, 12], [3, 2]);
const c = a.matmul(b);

console.log(await c.toArray()); // Float32Array [58, 64, 139, 154]
```

## Quick Start

```javascript
import { Tensor, noGrad, crossEntropy, trackTensors, init } from "torchic";

// Initialize with the JS Workers backend (or "wasm") and 4 worker threads
await init({ backend: "workers", threadCount: 4 });

// Create tensors
const x = Tensor.fromData([1, 2, 3, 4], [2, 2]);
const w = Tensor.randn([2, 2], true); // requires_grad=true
const b = Tensor.zeros([2], true);

// Forward pass (all synchronous-looking!)
const y = x.matmul(w).add(b);
const loss = y.sum();

// Backward pass
loss.backward();

// Read results (async only when reading data)
console.log("Loss:", await loss.item());
console.log("Gradient:", await w.grad.toArray());

// Example: using trackTensors to auto-dispose temporaries
await trackTensors(async () => {
  const temp = x.add(w);
  console.log(await temp.item());
});
```

## Training Example

```javascript
// Linear regression: y = 2x + 1
const x = Tensor.fromData([1, 2, 3, 4, 5], [5, 1]);
const y_true = Tensor.fromData([3, 5, 7, 9, 11], [5, 1]);

// Initialize parameters
let w = Tensor.randn([1, 1], true);
let b = Tensor.zeros([1], true);

const lr = 0.01;

for (let epoch = 0; epoch < 100; epoch++) {
  // Forward
  const y_pred = x.matmul(w).add(b);
  const loss = y_pred.sub(y_true).mul(y_pred.sub(y_true)).mean();

  // Backward
  loss.backward();

  // Update (disable autograd during parameter updates)
  await noGrad(async () => {
    if (w.grad) w.sub_(w.grad.mul(Tensor.fromData([lr], [1])));
    if (b.grad) b.sub_(b.grad.mul(Tensor.fromData([lr], [1])));
    // Zero gradients
    w.grad = null;
    b.grad = null;
  });

  if (epoch % 10 === 0) {
    console.log(`Epoch ${epoch}: Loss = ${await loss.item()}`);
  }
}

console.log("Final w:", await w.item()); // ~2.0
console.log("Final b:", await b.item()); // ~1.0
```

## API Reference

### Tensor Creation

```javascript
// From data
Tensor.fromData([1, 2, 3, 4], [2, 2]);

// Random initialization
Tensor.randn([128, 64], (requiresGrad = false));

// Zeros
Tensor.zeros([10, 10], (requiresGrad = false));
```

### Operations

**Math Operations** (all return new Tensors):

- `.add(other)` - Element-wise addition
- `.sub(other)` - Element-wise subtraction
- `.mul(other)` - Element-wise multiplication
- `.div(other)` - Element-wise division
- `.matmul(other)` - Matrix multiplication
- `.neg()` - Negation

**Slicing and Indexing**:

- `.slice(ranges)` - N-dimensional slicing, returns a tensor view. Example: `tensor.slice([[0,2],[1,4]])`
- `.set(indices, value)` - Set value at n-dimensional indices. Example: `tensor.set([i, j], value)`

**Activations**:

- `.relu()` - ReLU activation
- `.exp()` - Exponential
- `.log()` - Natural logarithm
- `.softmax(axis)` - Softmax activation

**Reductions**:

- `.sum(axis?, keepDim?)` - Sum reduction
- `.mean()` - Mean of all elements

**Shape Operations** (zero-copy):

- `.reshape(newShape)` - Reshape tensor
- `.transpose()` - Transpose 2D tensor

**Autograd**:

- `.backward()` - Compute gradients via backpropagation
- `noGrad(async () => {...})` - Disable gradient computation

**Data Access** (async):

- `await tensor.item()` - Read scalar value (first/only element)
- `await tensor.toArray()` - Read as Float32Array
- `tensor.slice(ranges)` - Get a view of a region (see above)
- `tensor.set(indices, value)` - Set value at indices (see above)

### In-Place Operations

Operations ending with `_` modify the tensor in place:

- `.add_(other)`, `.sub_(other)`, `.mul_(other)`, `.div_(other)`

**Warning**: In-place operations should only be used inside `noGrad()` blocks to avoid breaking the computation graph, when autograd is enabled.

### Static Methods

```javascript
// Cross-entropy loss (direct export)
import { crossEntropy } from "torchic";
const loss = crossEntropy(logits, target);

// Track and auto-dispose temporary tensors (direct export)
import { trackTensors } from "torchic";
await trackTensors(async () => {
  // ... create temporary tensors ...
});

// Initialize the backend + worker pool
await init({ backend: "workers", threadCount: 4 });
```

## Architecture

Frontend (main thread): a `Tensor` class holding shape/strides/ID, an autograd DAG builder, and a `Dispatcher` that forwards ops to whichever backend was chosen at `init`.

Backends all implement the same `Dispatcher` interface:

- `workers`: coordinator worker + N compute workers over a `SharedArrayBuffer`.
- `wasm`: same shape as `workers`; compute workers each instantiate the Rust WASM module against shared `WebAssembly.Memory`.
- `webgpu`: main-thread dispatcher, one `GPUBuffer` heap, WGSL compute shaders. No worker pool; GPU parallelism lives inside the shader.

Every backend sub-allocates its heap through a segregated free-list allocator with per-size-class LIFO buckets. NN workloads reuse identical tensor shapes each iteration, so most allocations recycle in `O(1)`. Reshape/transpose/slice rewrite strides without moving data, and freed views don't touch the parent's storage. Garbage collection uses `FinalizationRegistry`.

## Backends & Performance

Three interchangeable compute backends behind one dispatcher interface. Pick one at init:

```ts
await init({ backend: "workers", threadCount: 4 }); // JS on Web Workers
await init({ backend: "wasm",    threadCount: 4 }); // SIMD Rust → WebAssembly
await init({ backend: "webgpu"                  }); // WGSL compute shaders on the GPU
```

`workers` and `wasm` share the same architecture: a coordinator worker owns the shared memory + allocator, and N compute workers execute row-sliced kernels in parallel over one `SharedArrayBuffer`. Only the compute step differs. `webgpu` is single-dispatch from the main thread; GPU parallelism lives inside the shader (workgroups + threads), not in JS.

### `matmul` GFLOPS

Higher is better. `workers` and `wasm` at 8 threads (peak). `webgpu` is single-dispatch and thread-count-invariant.

| Backend | 128³ | 512³ | 1024³ | 2048³ |
| ---: | ---: | ---: | ---: | ---: |
| workers | 0.68 | 2.72 | 3.09 | 3.15 |
| wasm | 6.40 | 25.13 | **57.73** | 47.38 |
| webgpu | 1.10 | 66.69 | 354.08 | **310** |

Peak: 310 GFLOPS on WebGPU (2048³), 57 GFLOPS on WASM (1024³ @ 8t), 3.4 GFLOPS on Workers (1024³ @ 8t).

### Speedups vs Workers @ 8t

| Shape | wasm | webgpu |
| ---: | ---: | ---: |
| $128^3$ | 9.4× | 1.6× |
| $512^3$ | 9.2× | 24.5× |
| $1024^3$ | 18.7× | **114.6×** |
| $2048^3$ | 15.0× | **98.4×** |

Compound (`webgpu / workers`) hits ~100× at 1024³+.

### The numbers

Workers. Blocked matmul (BLOCK=32), row-parallel dispatch, two-phase SUM reduce, static work partitioning. Pure JavaScript, no SIMD. V8's TurboFan auto-vectorizes tight typed-array loops sometimes, but not reliably. Ceiling: ~1 GFLOPS per thread on this CPU.

WASM. Same architecture as workers, plus:

- SIMD128 (`+simd128`): `f32x4_add`/`mul`/`max`, `v128_load`/`store` on binary elementwise ops, reductions, fill, copy, matmul.
- 4×8 register-blocked matmul microkernel. Eight `f32x4` accumulators live in wasm locals; LLVM lowers them to CPU SIMD registers. Accumulator never touches memory during the k-loop.
- K-blocked A-panel packing. For each k-block of 256, pack `A[i:i+4, k:k+256]` contiguously into a scratch region so the microkernel's 4 A-loads per k-step hit adjacent bytes. Doubled 1024³ throughput.
- Shared `WebAssembly.Memory({ shared: true })` imported by the module. All workers share one SAB-backed linear memory; kernels operate at raw byte offsets, no copy across the JS/WASM boundary.
- `no_std` cdylib, LTO, `opt-level = 3`, `codegen-units = 1`, `panic = "abort"`. Zero runtime.

The 57 → 47 GFLOPS dip at 2048³ is L2 cache eviction. The 2048² working set doesn't fit in the 256 KB per-core L2 on this CPU. Needs a second layer of blocking + panel packing that isn't yet implemented.

WebGPU. 20+ WGSL compute shaders, one storage buffer heap sub-allocated by the same allocator as the other backends. Matmul is 16×16 tiled with workgroup shared memory and cooperative loads; barriers separate load and FMA passes. Elementwise ops are one thread per output element, 256-thread workgroups. Same `Dispatcher` interface, same `Tensor` frontend, no autograd changes.

A register-blocked WGSL variant (each thread computing a 4×4 output block, 64×64 workgroup output tile) regressed 2× from 16-way shared-memory bank conflicts on the packed-A reads. Reverted. Fixing it needs `vec4` loads, transposed thread mapping, and 128×128 tiles, the shape tinygrad and WebLLM's WGSL GEMM shaders use.

128³ is dispatch-overhead-dominated: ~200µs of encode + `mapAsync` roundtrip beats ~5µs of GPU compute, so WebGPU loses to WASM there.

On the 2070 Max-Q under Chrome's default power state, WebGPU holds ~310 GFLOPS steady on 2048³ (single-thread hot loop, 50 iterations, median). Chrome's `powerPreference: high-performance` hint is ignored on Windows ([crbug.com/369219127](https://crbug.com/369219127)), so the dGPU sits at ~50% clocks. NVIDIA Control Panel → "Prefer maximum performance" for chrome.exe roughly doubles this. Desktop GPUs are not power-managed this way.

### Test machine

| | |
| --- | --- |
| CPU | Intel Core i7-10750H (6 cores / 12 threads, 2.6 / 5.0 GHz) |
| GPU | NVIDIA RTX 2070 Max-Q |
| RAM | 16 GB |
| OS | Windows 11 |
| Browser | Microsoft Edge 142 |

## Future Improvements

- Better WebGPU matmul. `vec4<f32>` loads, transposed thread mapping to fix bank conflicts, 128×128 workgroup tiles with each thread computing 8×8 outputs. Target: 1-2 TFLOPS on 2070-class GPUs.
- Second-level cache blocking in WASM. Fix the 2048³ dip by blocking B-panels for L2, not just L1.
- SIMD polynomial `exp`/`log`/`tanh`. Currently scalar via `libm`. A 5th-order minimax poly would give 2-4× on softmax-heavy code.
- Broadcast fast paths for WASM/WebGPU elementwise. Currently the coordinator only packs shape/strides for `MATERIALIZE`; the same pattern would let broadcast add/sub/mul/div stay in the SIMD fast path.
- Backend-parameterized test suite. Run the existing kernel tests against all three backends automatically.

## Tests, benches, and demos

Layout under [tests/](tests/):

- `tests/unit/`: Vitest unit tests for kernels, allocator, tensor helpers, broadcast. Run with `npm test`.
- `tests/framework/`: DOM harness. `defineTest` / `defineBench` register suites; `mount()` renders.
- `tests/suites/`: browser-only suites: kernel parity (WASM ↔ Workers, WebGPU ↔ Workers) and a matmul GFLOPS bench across all three backends.
- `tests/demos/`: playground scripts (e.g. `makemore.ts`).

Open the interactive page:

```bash
npm run dev
# Open browser to http://localhost:5173/
```

Add a bench by dropping a file in `tests/suites/` and importing it from `tests/index.ts`:

```ts
import { defineBench } from "../framework/define";

defineBench(
  "My new bench",
  async (threads, { log }) => {
    log(`running with ${threads} threads`);
    // ... run some work ...
    return { threads, opsPerSec: 12345 };
  },
  [1, 2, 4, 8],
);
```

Each suite renders as N param buttons + Run all. Results fill a table row by row; log output streams into a collapsible panel.

## License

MIT