# 🔥 torchic

A toy neural network library that runs in the browser across three interchangeable compute backends: JS Web Workers, SIMD Rust compiled to WebAssembly, and WebGPU.

## Overview

Write sync-looking PyTorch-style code in JavaScript. Everything heavy runs off the main thread — on Web Workers for the CPU backends, on the GPU queue for WebGPU. Includes reverse-mode autograd, one shared-memory heap that all backends allocate against, and zero-copy tensor views.

### Key Features

- Three backends behind one `Dispatcher` interface. Peaks on the test laptop: **310 GFLOPS WebGPU / 57 GFLOPS WASM / 3.4 GFLOPS Workers** on `matmul`.
- Reverse-mode autograd over a dynamic computation graph.
- One `SharedArrayBuffer` heap shared across all workers, sub-allocated by a segregated free-list with O(1) block recycling.
- Zero-copy reshape and transpose — views only rewrite strides.
- Main thread never blocks. Kernels run on Web Workers or the GPU queue; only `.item()` / `.toArray()` await results.

## Installation

Literally uninstallable at the moment

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

torchic uses a **Frontend/Backend** architecture:

**Frontend (Main Thread)**:

- `Tensor` class: Lightweight metadata wrapper (shape, strides, ID)
- Autograd engine: Builds computation graph (DAG)
- Dispatcher: Serializes operations to backend

**Backend (Web Workers)**:

- Coordinator worker: Manages memory and task distribution
- Compute workers: Execute parallel math operations
- Memory allocator: Manages SharedArrayBuffer heap (malloc/free)
- Kernel library: Optimized math implementations

### Zero-Copy Memory Sharing

All tensors live in a single `SharedArrayBuffer` (default 256MB). Workers access data by offset, enabling:

- **Zero data transfer**: No copying between threads
- **View operations**: Reshape/transpose just modify strides
- **Parallel execution**: Multiple workers compute on same buffer

### Memory Management

- Automatic garbage collection via `FinalizationRegistry`.
- Segregated free-list allocator with per-size-class LIFO buckets — most tensor allocations recycle in O(1) since NN workloads reuse identical shapes each iteration.
- View tensors (reshape/transpose/slice) share memory with the parent; freed views don't touch the parent's storage.

## Use Cases

Just for fun, to teach myself how to work with shared-memory parallelism in JS, implementing and optimizing kernels in Rust/WASM as well as implementing automatic differentiation (autograd). Potentially useful for demos and visualizations, since everything runs client-side.

- 📚 **Education**: Learn autograd, strided arrays, and async dispatch
- 🎮 **Client-Side ML**: Train small models in the browser without backend
- 📊 **Visualization**: Real-time training visualization with React/Canvas
- 🧪 **Prototyping**: Quick experimentation with neural networks

## Backends & Performance

Three interchangeable compute backends behind one dispatcher interface. Pick one at init:

```ts
await init({ backend: "workers", threadCount: 4 }); // JS on Web Workers
await init({ backend: "wasm",    threadCount: 4 }); // SIMD Rust → WebAssembly
await init({ backend: "webgpu"                  }); // WGSL compute shaders on the GPU
```

`workers` and `wasm` share the same architecture — a coordinator worker owns the shared memory + allocator, and N compute workers execute row-sliced kernels in parallel over one `SharedArrayBuffer`. Only the compute step differs. `webgpu` is single-dispatch from the main thread; GPU parallelism lives inside the shader (workgroups + threads), not in JS.

### `matmul` GFLOPS

Higher is better. `workers` and `wasm` at 8 threads (peak). `webgpu` is single-dispatch and thread-count-invariant.

| Backend | 128³ | 512³ | 1024³ | 2048³ |
| ---: | ---: | ---: | ---: | ---: |
| workers | 0.68 | 2.72 | 3.09 | 3.15 |
| wasm | 6.40 | 25.13 | **57.73** | 47.38 |
| webgpu | 1.10 | 66.69 | 354.08 | **310** |

Peak: **310 GFLOPS on WebGPU** (2048³), **57 GFLOPS on WASM** (1024³ @ 8t), **3.4 GFLOPS on Workers** (1024³ @ 8t).

### Speedups vs Workers @ 8t

| Shape | wasm | webgpu |
| ---: | ---: | ---: |
| $128^3$ | 9.4× | 1.6× |
| $512^3$ | 9.2× | 24.5× |
| $1024^3$ | 18.7× | **114.6×** |
| $2048^3$ | 15.0× | **98.4×** |

Compound (`webgpu / workers`) hits ~100× at 1024³+.

### How each backend gets its numbers

**Workers.** Blocked matmul (BLOCK=32), row-parallel dispatch, two-phase SUM reduce, static work partitioning. Pure JavaScript, no SIMD access — V8's TurboFan sometimes auto-vectorizes tight typed-array loops but you can't rely on it. Ceiling: ~1 GFLOPS per thread on this CPU.

**WASM.** Same architecture as workers, plus:

- SIMD128 (`+simd128`) — `f32x4_add`/`mul`/`max`, `v128_load`/`store` on binary elementwise ops, reductions, fill, copy, matmul.
- 4×8 register-blocked matmul microkernel. Eight `f32x4` accumulators live in wasm locals; LLVM lowers them to CPU SIMD registers. Accumulator never touches memory during the k-loop.
- K-blocked A-panel packing. For each k-block of 256, pack `A[i:i+4, k:k+256]` contiguously into a scratch region so the microkernel's 4 A-loads per k-step hit adjacent bytes. Doubled 1024³ throughput on its own.
- Shared `WebAssembly.Memory({ shared: true })` imported by the module. All workers share one SAB-backed linear memory; kernels operate at raw byte offsets — no copy across the JS/WASM boundary.
- `no_std` cdylib, LTO, `opt-level = 3`, `codegen-units = 1`, `panic = "abort"`. Zero runtime.

**WebGPU.** 20+ WGSL compute shaders, one storage buffer heap sub-allocated by the same allocator as the other backends. Matmul is 16×16 tiled with workgroup shared memory and cooperative loads; barriers separate load and FMA passes. Elementwise ops are one thread per output element, 256-thread workgroups. Reused the same `Dispatcher` interface, same `Tensor` frontend — autograd required zero changes.

We tried a register-blocked WGSL variant (each thread computing 4×4 outputs, workgroup covering 64×64) — the WGSL analogue of what worked in Rust. It **regressed 2×** due to 16-way shared-memory bank conflicts on the packed-A reads. Reverted. Fixing it needs `vec4` loads + transposed thread mapping + 128×128 tiles, which is a research-scale problem — WebGPU GEMM kernels in tinygrad and WebLLM took engineer-months to tune.

### Notes on the numbers

**128³ is dispatch-overhead-dominated.** WebGPU actually loses to WASM here — ~200µs of encode + `mapAsync` roundtrip beats ~5µs of GPU compute. Real workloads don't touch 128³ often.

**WASM's dip at 2048³** (57 → 47 GFLOPS) is L2 cache eviction. The 2048² working set doesn't fit in the 256 KB per-core L2 on this CPU. Fixing it needs a second layer of blocking + panel packing that we haven't done.

**WebGPU on 2070 Max-Q under Chrome default power state.** ~310 GFLOPS steady on 2048³ (single-thread hot-loop measurements, 50 iterations, median). Chrome's `powerPreference: high-performance` hint is currently ignored on Windows ([crbug.com/369219127](https://crbug.com/369219127)), so the dGPU sits at ~50% clocks. Setting NVIDIA Control Panel → "Prefer maximum performance" for chrome.exe unlocks the full clock and roughly doubles this number. Desktop GPUs don't have this asymmetry.

### Test machine

| | |
| --- | --- |
| CPU | Intel Core i7-10750H (6 cores / 12 threads, 2.6 / 5.0 GHz) |
| GPU | NVIDIA RTX 2070 Max-Q |
| RAM | 16 GB |
| OS | Windows 11 |
| Browser | Microsoft Edge 142 |

## Future Improvements

- **Better WebGPU matmul.** `vec4<f32>` loads, transposed thread mapping to fix bank conflicts, 128×128 workgroup tiles with each thread computing 8×8 outputs. Target: 1-2 TFLOPS on 2070-class GPUs.
- **Second-level cache blocking in WASM.** Fix the 2048³ dip by blocking B-panels for L2, not just L1.
- **SIMD polynomial `exp`/`log`/`tanh`.** Currently scalar via `libm`. A 5th-order minimax poly would give 2-4× on softmax-heavy code.
- **Broadcast fast paths for WASM/WebGPU elementwise.** Currently the coordinator only packs shape/strides for `MATERIALIZE`; the same pattern would let broadcast add/sub/mul/div stay in the SIMD fast path.
- **Backend-parameterized test suite.** Run the existing kernel tests against all three backends automatically.

## Tests, benches, and demos

The [tests/](tests/) directory is organized as follows:

- `tests/unit/` – Vitest unit tests (kernels, memory allocator, tensor helpers, broadcast semantics). Run with `npm test`.
- `tests/framework/` – Tiny UI harness: `defineTest` / `defineBench` register suites; `mount()` generates the DOM. CSS lives in [`framework/styles.css`](tests/framework/styles.css).
- `tests/suites/` – Browser-only suites registered against the framework. Currently: WASM ↔ Workers kernel parity and a matmul GFLOPS bench across both backends.
- `tests/demos/` – Playground scripts (e.g. `makemore.ts`).

To open the interactive page:

```bash
npm run dev
# Open browser to http://localhost:5173/
```

Add a new bench or test by dropping a file in `tests/suites/` and importing it from `tests/index.ts`:

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

The framework generates buttons per param + a "Run all", collects results into a table, and streams log output to a collapsible panel.

## License

MIT