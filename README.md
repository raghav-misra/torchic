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
import { Tensor, noGrad, crossEntropy, trackTensors } from 'torchic';

// Initialize with 4 worker threads
await Tensor.init(4);

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
console.log('Loss:', await loss.item());
console.log('Gradient:', await w.grad.toArray());

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

console.log('Final w:', await w.item()); // ~2.0
console.log('Final b:', await b.item()); // ~1.0
```

## API Reference

### Tensor Creation

```javascript
// From data
Tensor.fromData([1, 2, 3, 4], [2, 2])

// Random initialization
Tensor.randn([128, 64], requiresGrad = false)

// Zeros
Tensor.zeros([10, 10], requiresGrad = false)
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
import { crossEntropy } from 'torchic';
const loss = crossEntropy(logits, target);

// Track and auto-dispose temporary tensors (direct export)
import { trackTensors } from 'torchic';
await trackTensors(async () => {
  // ... create temporary tensors ...
});

// Initialize worker pool
await Tensor.init(numThreads = 4)
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

Just for fun, to teach myself how to work with shared-memory parallelism in JS, as well as implementing automatic differentiation (autograd). Potentially useful for demos and visualizations, since everything runs client-side.

- 📚 **Education**: Learn autograd, strided arrays, and async dispatch
- 🎮 **Client-Side ML**: Train small models in the browser without backend
- 📊 **Visualization**: Real-time training visualization with React/Canvas
- 🧪 **Prototyping**: Quick experimentation with neural networks

## Performance Notes

- **CPU-focused**: Optimized for multi-threaded CPU execution
- **Best for**: Small to medium models (MLPs, small transformers)
- **Limitations**: Not a replacement for GPU-accelerated libraries
- **Browser support**: Requires SharedArrayBuffer (COOP/COEP headers)

## Project Structure

```
torchic/
├── src/
│   ├── index.ts           # Main exports
│   ├── engine/
│   │   └── tensor.ts      # Tensor class & autograd
│   ├── backend/
│   │   ├── dispatcher.ts  # Main thread ↔ Worker communication
│   │   ├── memory.ts      # SharedArrayBuffer heap manager
│   │   └── worker.ts      # Worker thread logic
│   └── kernels/           # Math implementations
│       ├── elementwise.ts # Element-wise operations
│       ├── matmul.ts      # Matrix multiplication
│       ├── reductions.ts  # Sum, mean operations
│       └── transpose.ts   # Transpose kernel
├── tests/
│   └── test.ts            # Comprehensive test suite
└── DESIGN.md              # Architecture documentation
```

## Examples in Test Suite

The `tests/test.ts` file contains examples for:
- Basic operations and broadcasting
- Autograd and gradient checking
- Matrix multiplication with gradients
- MLP layer forward/backward
- Linear regression with adaptive learning rate
- Multivariate regression
- Cross-entropy loss
- Softmax layer
- Zero-copy reshape and transpose

Run the test page:
```bash
npm run dev
# Open browser to localhost:5173/tests/
```

## License

MIT

---

**Note**: This is an educational project demonstrating neural network fundamentals and async compute patterns. For production use cases to run on the web, consider established libraries like TensorFlow.js or ONNX Runtime Web.
