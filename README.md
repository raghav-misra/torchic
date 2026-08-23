# 🔥 torchic

A toy neural-network library that runs in the browser across three
interchangeable compute backends: JS Web Workers, SIMD Rust compiled to
WebAssembly, and WebGPU. PyTorch-style API, reverse-mode autograd, one shared
tensor heap, zero-copy views.

## Highlights

- **Serves [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS
  end-to-end in-browser.** 82M-param model, real HF safetensors weights,
  Float32 throughout, 24 kHz output. See
  [`tests/demos/kokoro/`](tests/demos/kokoro/) and the "Kokoro: end-to-end
  synthesis" freeform bench in the interactive test page.
- **Three interchangeable backends** behind one `Dispatcher` interface. Pick
  at `init`.
- **Peak matmul: 310 GFLOPS on WebGPU, 57 GFLOPS on WASM** (2070 Max-Q / i7).
  See [Performance](#performance).
- **Kernels across all backends:** matmul, bmm, Conv1d/ConvTranspose1d (incl.
  grouped/depthwise), softmax, sum, sum-axis, transpose, embedding, concat,
  pad, `relu`/`gelu`/`tanh`/`sigmoid`/`silu`/`leaky_relu`/`sqrt`/`rsqrt`/
  `sin`/`cos`/`exp`/`log`.
- **nn:** `Linear`, `LinearNorm`, `Embedding`, `Sequential`, `LayerNorm`,
  `GroupNorm`, `InstanceNorm1d`, `MultiHeadAttention`,
  `TransformerEncoderLayer`, `Conv1d`, `ConvTranspose1d`, `LSTMCell`,
  `BiLSTM`, `Snake1D`. Loads HF safetensors (F32/BF16/F16) with `weight_norm`
  fusion on the fly.
- **dsp:** `stft` / `istft` (non-pow-2 FFT supported), `hannWindow`.

## Quick start

```ts
import { Tensor, init } from "torchic";

await init({ backend: "webgpu", memorySizeMB: 512 });

const a = Tensor.fromData([1, 2, 3, 4, 5, 6], [2, 3]);
const b = Tensor.fromData([7, 8, 9, 10, 11, 12], [3, 2]);
console.log(await a.matmul(b).toArray()); // Float32Array [58, 64, 139, 154]
```

Autograd is on by default; wrap inference in `noGrad`, or call `model.eval()`
which does it globally.

## Kokoro TTS demo

```ts
import { init, noGrad, nn } from "torchic";
import { Kokoro } from "./tests/demos/kokoro";

await init({ backend: "webgpu", memorySizeMB: 1536 });

const model = new Kokoro();
model.eval();

const checkpoint = nn.parseSafetensors(await (await fetch("kokoro-v1_0.safetensors")).arrayBuffer());
model.load_safetensors(checkpoint);

const voice = nn.parseSafetensors(await (await fetch("af_bella.safetensors")).arrayBuffer())["voice"];
const refS = Tensor.fromData(
  Array.from(voice.data.subarray(/* per-phoneme-count row */)), [1, 256],
);

// "Hello world" as Kokoro phoneme IDs (see KOKORO_CONFIG.vocab).
const ids = [0, 50, 156, 86, 54, 57, 135, 16, 65, 156, 87, 123, 54, 46, 0];
const inputIds = Tensor.fromData(ids, [1, ids.length]);
const { audio } = await noGrad(() => model.forward(inputIds, refS));
// audio: Float32Array PCM at 24 kHz. Route through Web Audio.
```

Convert the reference `.pth` checkpoint with
[`scripts/convert-kokoro-checkpoint.py`](scripts/convert-kokoro-checkpoint.py).

## Installation

```bash
npm install torchic
```

Requires:

1. Vite-style asset imports and the `new Worker(new URL(...), { type: "module" })` pattern.
2. Cross-origin isolation for `SharedArrayBuffer` — dev/prod server must send:

   ```
   Cross-Origin-Opener-Policy: same-origin
   Cross-Origin-Embedder-Policy: require-corp
   ```

3. Chromium 113+ (or Safari 18+ with WebGPU flag) for the `webgpu` backend.
   `workers` and `wasm` run anywhere modern.

Vite config:

```ts
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

## Architecture

Frontend: a `Tensor` class holding shape/strides/id, an autograd DAG builder,
and a `Dispatcher` that forwards ops to whichever backend was chosen at
`init`.

- `workers`: coordinator worker + N compute workers over a `SharedArrayBuffer`.
- `wasm`: same shape as `workers`; compute workers each instantiate the Rust
  WASM module against shared `WebAssembly.Memory`.
- `webgpu`: main-thread dispatcher, one `GPUBuffer` heap, WGSL compute shaders.
  No worker pool; GPU parallelism lives inside the shader.

Every backend sub-allocates its heap through a segregated free-list allocator
with per-size-class LIFO buckets. Reshape/transpose/slice rewrite strides
without moving data; freed views don't touch the parent's storage. GC uses
`FinalizationRegistry`, with explicit `.dispose()` available for hot paths.

## Performance

### `matmul` GFLOPS

`workers` and `wasm` at 8 threads (peak). `webgpu` is single-dispatch and
thread-count-invariant.

| Backend | 128³ | 512³ | 1024³ | 2048³ |
| ---: | ---: | ---: | ---: | ---: |
| workers | 0.68 | 2.72 | 3.09 | 3.15 |
| wasm | 6.40 | 25.13 | **57.73** | 47.38 |
| webgpu | 1.10 | 66.69 | 354.08 | **310** |

Speedups vs `workers @ 8t`:

| Shape | wasm | webgpu |
| ---: | ---: | ---: |
| 128³ | 9.4× | 1.6× |
| 512³ | 9.2× | 24.5× |
| 1024³ | 18.7× | **114.6×** |
| 2048³ | 15.0× | **98.4×** |

### What each backend actually does

**Workers.** Blocked matmul (BLOCK=32), row-parallel dispatch, two-phase SUM
reduce, static work partitioning. Pure JS. Ceiling ~1 GFLOPS/thread.

**WASM.** Rust `cdylib`, `no_std`, LTO, SIMD128 (`+simd128`). 4×8
register-blocked matmul microkernel with eight `f32x4` accumulators — LLVM
lowers them to CPU SIMD registers, so the accumulator never touches memory
during the k-loop. K-blocked A-panel packing keeps the microkernel's A-loads
cache-hot. Shared `WebAssembly.Memory` across workers, no JS/WASM copy at op
boundaries. The 2048³ dip is L2 eviction — needs a second layer of blocking.

**WebGPU.** 20+ WGSL compute shaders, one storage buffer heap. Matmul is
16×16 tiled with workgroup shared memory; barriers separate load and FMA
passes. Elementwise ops are one thread per output element, 256-thread
workgroups. A register-blocked variant regressed 2× from shared-memory bank
conflicts; fixing it needs `vec4` loads + 128×128 tiles (tinygrad / WebLLM
shape). 128³ is dispatch-overhead-dominated: encode + `mapAsync` roundtrip
beats the compute.

### Test machine

| | |
| --- | --- |
| CPU | Intel Core i7-10750H (6 cores / 12 threads) |
| GPU | NVIDIA RTX 2070 Max-Q |
| RAM | 16 GB |
| OS | Windows 11 |
| Browser | Microsoft Edge 142 |

## Future

- Better WebGPU matmul: `vec4<f32>` loads, transposed thread mapping, 128×128
  workgroup tiles. Target 1–2 TFLOPS on 2070-class GPUs.
- Second-level cache blocking in WASM.
- SIMD polynomial `exp` / `log` / `tanh`.
- Broadcast fast paths for WASM/WebGPU elementwise.
- Fused MHA on WebGPU; im2col Conv1d on WebGPU.
- Backend-parameterized test suite.

## Tests, benches, demos

```
tests/
├── unit/         # vitest — kernels, allocator, tensor helpers, broadcast
├── framework/    # defineTest / defineBench / defineFreeform + DOM harness
├── suites/       # cross-backend parity + matmul GFLOPS bench
└── demos/        # kokoro/ (TTS), makemore.ts (MLP)
```

Interactive:

```bash
npm run dev  # then open http://localhost:5173/
```

Headless (via puppeteer):

```bash
npm run bench -- wasm      # run WASM parity suite
npm run bench -- webgpu    # WebGPU parity
npm run bench -- kokoro    # Kokoro skeleton param count
npm run bench -- --list    # see all suites
```

Unit tests:

```bash
npm test
```

## License

MIT
