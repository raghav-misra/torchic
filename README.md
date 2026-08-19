# 🔥 torchic

A toy neural network library for the browser that runs on CPU with async execution using SharedArrayBuffer and Web Workers.

## Overview

**torchic** lets you write synchronous-looking neural network code in JavaScript while all the heavy computation happens asynchronously on background threads, keeping your UI responsive. It features automatic differentiation (autograd), zero-copy tensor operations, and multi-threaded CPU execution.

### Key Features

- Write standard PyTorch-esque code (`z = x.matmul(w).add(b)`)
- Math runs on Web Workers, never blocks the main thread
- Automatic differentiation with reverse-mode backpropagation
- SharedArrayBuffer means no data copying between threads
- Custom allocator with automatic garbage collection
- Zero-copy reshape and transpose operations

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

- Automatic garbage collection using `FinalizationRegistry`
- First-fit allocator with coalescing for heap management
- View tensors (reshape/transpose) share memory with parent tensors

## Use Cases

Just for fun, to teach myself how to work with shared-memory parallelism in JS, implementing and optimizing kernels in Rust/WASM as well as implementing automatic differentiation (autograd). Potentially useful for demos and visualizations, since everything runs client-side.

- 📚 **Education**: Learn autograd, strided arrays, and async dispatch
- 🎮 **Client-Side ML**: Train small models in the browser without backend
- 📊 **Visualization**: Real-time training visualization with React/Canvas
- 🧪 **Prototyping**: Quick experimentation with neural networks

## Backends & Performance

torchic ships two interchangeable compute backends behind the same `Dispatcher` interface. Select one at init time:

```ts
await init({ backend: "workers", threadCount: 4 }); // hand-written JavaScript kernels
await init({ backend: "wasm",    threadCount: 4 }); // SIMD Rust compiled to WebAssembly
```

Both backends use the **same architecture**: a coordinator worker owns the shared memory + allocator, and `N` compute workers execute row-sliced kernels in parallel over a single `SharedArrayBuffer`. Only the compute step differs.

- **CPU-focused**: multi-threaded execution, no GPU
- **Best for**: small to medium models (MLPs, small transformers)
- **Not a replacement**: for GPU-accelerated libraries on large models
- **Browser requirements**: SharedArrayBuffer (COOP/COEP headers)

### `matmul` – Workers backend

Blocked (BLOCK=32) JavaScript kernels dispatched across N worker threads. Values are medians over 5-7 timed trials after warmup.

| Thread Count |     |         Shape (A: MxK, B: KxN) | Median (ms) | GFLOPS |
| -----------: | :-: | -----------------------------: | ----------: | -----: |
|            1 |     |    $128 \times 128 \times 128$ |        6.70 |  0.626 |
|              |     |    $512 \times 512 \times 512$ |      370.41 |  0.725 |
|              |     | $1024 \times 1024 \times 1024$ |     3009.09 |  0.714 |
|            2 |     |    $128 \times 128 \times 128$ |        4.03 |  1.042 |
|              |     |    $512 \times 512 \times 512$ |      200.55 |  1.339 |
|              |     | $1024 \times 1024 \times 1024$ |     1537.07 |  1.397 |
|            4 |     |    $128 \times 128 \times 128$ |        3.43 |  1.221 |
|              |     |    $512 \times 512 \times 512$ |      105.95 |  2.533 |
|              |     | $1024 \times 1024 \times 1024$ |      933.26 |  2.301 |
|            8 |     |    $128 \times 128 \times 128$ |        2.13 |  1.969 |
|              |     |    $512 \times 512 \times 512$ |       84.28 |  3.185 |
|              |     | $1024 \times 1024 \times 1024$ |      639.23 |  3.360 |

### `matmul` – Rust/WASM backend

Rust kernels compiled to WebAssembly with `+simd128` and shared memory (`--import-memory --shared-memory`). Same coordinator + N compute worker pipeline; the compute workers instantiate the same compiled module against one shared `WebAssembly.Memory`.

| Thread Count |     |         Shape (A: MxK, B: KxN) | Median (ms) | GFLOPS |
| -----------: | :-: | -----------------------------: | ----------: | -----: |
|            1 |     |    $128 \times 128 \times 128$ |        0.70 |  6.035 |
|              |     |    $512 \times 512 \times 512$ |       39.77 |  6.750 |
|              |     | $1024 \times 1024 \times 1024$ |      333.55 |  6.438 |
|            2 |     |    $128 \times 128 \times 128$ |        0.58 |  7.170 |
|              |     |    $512 \times 512 \times 512$ |       20.31 | 13.214 |
|              |     | $1024 \times 1024 \times 1024$ |      191.55 | 11.211 |
|            4 |     |    $128 \times 128 \times 128$ |        0.55 |  7.696 |
|              |     |    $512 \times 512 \times 512$ |       11.37 | 23.609 |
|              |     | $1024 \times 1024 \times 1024$ |       96.70 | 22.208 |
|            8 |     |    $128 \times 128 \times 128$ |        0.56 |  7.557 |
|              |     |    $512 \times 512 \times 512$ |        9.76 | 27.504 |
|              |     | $1024 \times 1024 \times 1024$ |       80.80 | 26.576 |

### Comparative analysis

WASM is **~9× faster than the workers backend at the same thread count** across all sizes. Single-thread WASM already beats 8-thread workers on all sizes. On 1024³ the peak is 27 GFLOPS on WASM vs 3.4 GFLOPS on workers.

Speedup ratios (`wasm / workers`, same thread count):

|        Shape |    1t |    2t |    4t |    8t |
| -----------: | ----: | ----: | ----: | ----: |
|    $128^3$   | 9.63× | 6.88× | 6.30× | 3.84× |
|    $512^3$   | 9.31× | 9.87× | 9.32× | 8.63× |
|    $1024^3$  | 9.02× | 8.02× | 9.65× | 7.91× |

**Where the gap comes from.** JS engines have no SIMD API – the `SIMD.js` proposal was withdrawn from ECMAScript in 2019 in favor of WebAssembly SIMD. V8's TurboFan does opportunistic auto-vectorization on tight typed-array loops sometimes, but you can't rely on it. Every workers-backend `f32 add` runs one lane at a time. Every wasm-backend `f32 add` runs four (`f32x4_add`). Multi-threading is symmetric across the two backends, so the wasm win is entirely per-thread work.

#### Workers-side optimizations

- **Blocked matmul** (BLOCK=32) to keep the inner block in L1 across k-iterations.
- **Row-parallel dispatch** – each worker owns a contiguous row range of the output, avoiding cache-line contention on writes.
- **Two-phase SUM reduce** – partial sums per worker written to per-worker scratch slots, then reduced by worker 0. No cross-thread synchronization during the partial phase.
- **Segregated free-list allocator** with per-size-class LIFO buckets. Because NN workloads reuse identical tensor shapes each iteration, most allocations recycle a block in O(1).
- **Static work partitioning** – no runtime work-stealing or queueing during a kernel.
- **Zero-copy views**: reshape/transpose only rewrite strides; the data stays put in the shared buffer.

#### Rust/WASM-side optimizations

Everything above, plus:

- **SIMD128 target feature** (`+simd128`) – `f32x4_add`, `f32x4_mul`, `f32x4_max`, `v128_load`/`store` on the hot paths for all binary elementwise ops, reductions, fill, copy, and matmul.
- **Register-blocked 4×8 matmul microkernel.** Eight `f32x4` accumulators live in wasm locals (which LLVM lowers to CPU SIMD registers), so the accumulator never touches memory during the k-loop. Amortizes A/B load cost across 32 output floats per pass.
- **8 independent FMA chains** in the microkernel – 4 output rows × 2 output col-lanes – giving modern CPUs the parallel dependency chains they need to actually issue 2-4 SIMD ops per cycle. This is what pushed matmul from ~10 GFLOPS to ~27 GFLOPS at 8 threads.
- **Shared `WebAssembly.Memory({ shared: true })`** – the module imports memory rather than declaring its own, so the coordinator + N compute workers all share one `SharedArrayBuffer`-backed linear memory. Kernels read/write tensor bytes directly at their JS-assigned byte offsets – no copy, no serialization across the JS/WASM boundary.
- **`no_std` cdylib, LTO, `opt-level = 3`, `codegen-units = 1`, `panic = "abort"`** – maximum inlining, no runtime, minimal binary.

#### Scaling behavior

Both backends scale roughly linearly from 1t → 4t, then flatten between 4t → 8t. The test machine has 6 physical cores + 12 logical threads. At 4 threads we still have room; at 8 threads two logical threads share each physical core's SIMD units and cache. On the largest workload (1024³), the wasm backend additionally starts hitting main-memory bandwidth: B is streamed through cache each output tile, and DRAM caps the win.

The 128³ case underperforms across the board because dispatch overhead (worker messaging, task promise resolution) dominates the actual compute. This is a benchmark artifact – real training loops don't touch 128³ much.

### Test machine / environment

| Field                            | Value                                                |
| -------------------------------- | ---------------------------------------------------- |
| CPU model                        | Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz (2.59 GHz) |
| Physical cores / Logical threads | 6 core / 12 logical processors                       |
| Base / boost frequency           | 2.60 GHz (base) / 5.00 GHz (boost)                   |
| RAM                              | 16.0 GB (15.8 GB usable)                             |
| OS                               | Windows 11 Version 25H2 (Build 26200.7171)           |
| Browser (name + version)         | Microsoft Edge 142.0.3595.94                         |

## Future Improvements

- **WebGPU backend:** GPU backend via WebGPU for large kernels and model training where GPU parallelism and memory bandwidth dominate.
- **Further wasm matmul tuning:** K-blocking + A-panel packing to cure the 4t → 8t stall on very large matmuls; SIMD polynomial `exp`/`log`/`tanh` for softmax-heavy workloads.
- **Broadcast fast paths in wasm:** currently the coordinator only packs shape/strides for `MATERIALIZE`; the same pattern would let broadcast add/sub/mul/div stay in the SIMD fast path instead of throwing.
- **Backend-parameterized test suite:** run the existing kernel tests against both backends automatically.
- **Benchmarking & profiling:** per-worker instrumentation, memory-bandwidth measurements, and automated perf tests to guide the next optimization.

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