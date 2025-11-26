# `torchic`

## Purpose

Bored and wanted to make something kind of cool. Minimal real-world utility for this. torchic allows users to write standard, synchronous-looking code in JS to express neural networks and perform operations across tensors (e.g., `z = x.matmul(y)`), while the actual computation happens asynchronously on background threads using `SharedArrayBuffer`, preventing UI freezes.

Ideally this can be used to create a fun toy neural network designer and executor web UI lol.


**Utility & Use Cases:**

  * **Educational:** A readable reference implementation of Autograd, Strided Arrays, and Asynchronous Dispatch.
  * **Client-Side Training:** Training small models (MLPs, baby Transformers) directly in the browser without a Python backend.
  * **Visualization:** Since the training loop runs in JS, it can be tightly coupled with React/Canvas for real-time visualization of weights and gradients during training.


## Component Summary

The system is divided into the **Frontend** (Main Thread) and the **Backend** (Worker Threads).

1.  **Frontend:**

      * **Tensor:** The primary data structure. Lightweight wrapper holding metadata (shape, strides) and a pointer to backend memory.
      * **Autograd Engine:** Builds the DAG (Directed Acyclic Graph) of operations for backpropagation.
      * **Dispatcher:** Serializes operations into commands and sends them to the backend.

2.  **Backend (Compute):**

      * **WorkerPool:** A cluster of Web Workers that execute math operations.
      * **MemoryAllocator:** Manages the `SharedArrayBuffer` heap, simulating `malloc`/`free`.
      * **Kernel Ops:** The actual math implementations (MatMul, ReLU, etc.).

## API Surface

The user interacts exclusively with the `torchic` global and `Tensor` instances.

### The `Tensor` Class

A handle to data living in the backend. Operations on Tensors return new Tensors immediately (lazy execution).

  * **Properties:**

      * `shape`: `number[]` (e.g., `[32, 784]`)
      * `requiresGrad`: `boolean`
      * `grad`: `Tensor | null`
      * `device`: `'cpu' | 'webgl'` (Future proofing)

  * **Methods:**

      * **Math:** `.add()`, `.sub()`, `.mul()`, `.matmul()`, `.relu()`, `.exp()`. All return a new `Tensor`.
      * **Reduction:** `.sum()`, `.mean()`.
      * **Backprop:** `.backward()` (Triggers the DAG traversal).
      * **Data Access:** `.item()` or `.toArray()`. **Crucial:** These are `async` and return a Promise. This is the only time the main thread waits for the backend.

### The `torchic` Namespace

  * `torchic.init({ threads: number })`: Initializes the SharedBuffer and Workers.
  * `torchic.tensor(data)`: Creates a tensor from a JS array.
  * `torchic.randn(shape)`: Gaussian initialization.
  * `torchic.zeros(shape)`: Zero initialization.
  * `torchic.noGrad(fn)`: Context manager to disable graph building (for optimization steps).

## Internal APIs

These components operate behind the scenes to enable the "User in Controller / Math in Worker" architecture.

### Dispatcher

The bridge between the clean API and the messy threading.

  * **Role:** Maintains a queue of operations.
  * **Protocol:** Uses `postMessage` to send lightweight "Command Objects" to workers.
      * *Command Structure:* `{ op: "MATMUL", inputIds: [uid1, uid2], outId: uid3, shape: [...] }`
  * **Sync Mechanism:** When the user calls `await tensor.item()`, the Dispatcher attaches a unique Request ID to the command and listens for a specific "Done" message from the worker.

### Memory Allocator

Since we use a flat `SharedArrayBuffer`, we need a custom memory manager (a simplified C-style `malloc`).

  * **Location:** Lives in the **Worker**. The Main thread only knows "Tensor ID 5", the Worker knows "Tensor ID 5 starts at byte 1024."
  * **Strategy:** For this project, a **Simple Bump Allocator** with a "Reset" capability is sufficient.
      * `allocate(size)`: Returns an offset index and increments the pointer.
      * `free()`: (Advanced) Marks blocks as reusable.

### 4.3 The Kernel Library

The actual math logic.

  * **Optimization:** Uses 1D array loop logic.
  * **Parallelism:** For operations like `matmul`, the Kernel calculates how to split rows across multiple workers (sharding) based on the input shape.

-----

## 5\. Codebase Structure

```text
torchic/
├── src/
│   ├── index.ts           # Entry point (exports torchic)
│   ├── engine/
│   │   ├── tensor.ts      # Tensor class & Autograd logic
│   │   ├── autograd.ts    # GraphNode and topological sort
│   │   └── ops.ts         # Forward/Backward op definitions
│   │
│   ├── backend/
│   │   ├── dispatcher.ts  # Main thread -> Worker communication
│   │   ├── memory.ts      # Heap management (SharedArrayBuffer)
│   │   └── worker.ts      # The code running inside the Web Worker
│   │
│   └── kernels/           # Raw Math implementations
│       ├── matmul.ts
│       ├── elementwise.ts
│       └── reduce.ts
│
├── examples/
│   ├── mnist_mlp.js       # End-to-end training demo
│   └── sanity_check.js    # Gradient checking
```

-----

## 6\. Examples

### Example 1: The "Lazy" Execution

```javascript
import { init, randn } from 'torchic';

await torchic.init();

// 1. Queue creation commands (Instant)
let a = torchic.randn([128, 128]);
let b = torch.randn([128, 128]);

// 2. Queue math command (Instant)
let c = a.matmul(b);

// 3. User does other UI work here...

// 4. Synchronization (Blocks until Worker finishes)
let result = await c.toArray();
console.log(result);
```

### Example 2: Training Loop

```javascript
// ... init code ...

for (let i = 0; i < 100; i++) {
    // Forward (All synchronous-looking)
    let y = x.matmul(w).add(b);
    let loss = y.sub(target).pow(2).mean();
    
    // Backward
    // Since 'loss' is just a metadata shell, we can call backward immediately.
    // The backend handles the dependency order.
    loss.backward(); 
    
    // Update
    torchic.noGrad(() => {
        w = w.sub(w.grad.mul(lr));
    });
    
    // Log occasionally (Async)
    if (i % 10 === 0) {
        console.log(`Loss: ${await loss.item()}`);
    }
}
```

-----

## Component Details & Implementation Guide

### The Tensor ID System

To avoid passing data back and forth, Tensors are just references.

  * **Implementation:**
      * Use a static counter or UUID for every `new Tensor()`.
      * Main Thread: `Tensor { id: 1, shape: [2,2] }`
      * Worker Thread: `Map<id, Float32ArrayView>`
  * **Challenge:** Garbage Collection. If the JS `Tensor` object is GC'd, we need to free the memory in the Worker.
      * *Solution (V2):* Use `FinalizationRegistry` to send a `FREE` command to the worker when the JS object dies.

### Dispatcher Protocol

The dispatcher needs to differentiate between "Fire and Forget" (math) and "Read" (IO).

```typescript
// types.ts
type Command = 
  | { type: 'ALLOC', id: string, shape: number[] }
  | { type: 'MATMUL', a_id: string, b_id: string, out_id: string }
  | { type: 'READ', id: string, req_id: string }; // req_id used to resolve Promise
```

### Autograd `backward()`

Adapt the `micrograd` approach.

1.  **Graph Construction:** When `c = a.add(b)` is called:
      * `c` is created.
      * `c.op = "add"`
      * `c.prev = [a, b]`
2.  **Topological Sort:**
      * Implement a recursive `buildTopo(node)` function.
3.  **Execution:**
      * Iterate the sorted nodes in reverse.
      * Call `node.backward()`.
      * *Crucially:* The `backward` function just queues *more* math operations to the backend. It does not calculate gradients in JS.
      * *Example:* `add_backward` queues a command to copy `out.grad` to `a.grad` and `b.grad`.

### Memory Layout (The Shared Buffer)

  * **Initialization:** `new SharedArrayBuffer(1024 * 1024 * 256)` (256MB).
  * **Structure:**
      * The buffer is just a sea of bytes.
      * The `MemoryAllocator` class maintains a pointer `currentOffset`.
      * When `ALLOC` is received: `offset = currentOffset; currentOffset += size`.
      * It returns `new Float32Array(buffer, offset, size)`.