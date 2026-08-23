# torchic → Kokoro TTS: model-serving roadmap

Living design doc. Update as reality collides with the plan.

## North star

Load the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS checkpoint in
the browser and stream synthesized audio for arbitrary English text, running
entirely on torchic. Target: RTF < 1.0 on a modern laptop with WebGPU, real-time
audio out via AudioWorklet.

Success looks like:

```ts
import { init } from "torchic";
import { Kokoro } from "torchic/models/kokoro";

await init({ backend: "webgpu", memorySizeMB: 512 });
const model = await Kokoro.fromPretrained("hexgrad/Kokoro-82M");
const audio = await model.synthesize("Hello, world.", { voice: "af_bella" });
audio.play();
```

## What we have today

Building blocks that already work in the library:

- [x] Tensor + autograd graph, `Tensor.zeros/randn/fromData`, `.reshape([-1])`
- [x] Three backends (Workers, WASM/SIMD, WebGPU) behind one dispatcher
- [x] `nn.Module` with `param`/`buffer`/`child`/`childList`, `state_dict`,
      `load_state_dict`, `train`/`eval`, `parameters()`
- [x] `nn.Linear` (N-D input, matmuls over last dim), `nn.Embedding`,
      `nn.Sequential`, `nn.LayerNorm` (composed)
- [x] `nn.MultiHeadAttention`, `nn.TransformerEncoderLayer`,
      `sinusoidalPositionalEncoding`
- [x] `nn.functional`: `relu`, `tanh`, `gelu`, `sigmoid`, `leaky_relu`, `silu`, `softmax`
- [x] `optim.SGD`
- [x] Broadcast materialize path on all three backends
- [x] Elementwise, matmul, transpose, softmax, sum, sum-axis, embedding fwd+bwd
- [x] `sqrt` / `rsqrt` primitives (Workers, WASM, WebGPU)
- [x] `gelu` primitive (Workers, WASM, WebGPU — tanh approximation used by BERT / Kokoro)
- [x] `sigmoid` primitive (Workers, WASM, WebGPU)
- [x] `leaky_relu` / `silu` primitives (Workers, WASM, WebGPU)
- [x] N-D `Tensor.transpose(dim0, dim1)` — zero-copy stride swap
- [x] Batched matmul `Tensor.bmm()` — Workers, WASM, WebGPU
- [x] `Conv1d` / `ConvTranspose1d` fwd — Workers, WASM, WebGPU (dilation +
      padding + stride, matches PyTorch layout)
- [x] `Tensor.reshape([-1])` auto-materializes non-contiguous inputs
- [x] Headless bench harness (vite + puppeteer) so all this can be exercised
      from CLI without opening a browser tab

## Gap analysis

Kokoro's public architecture is StyleTTS 2 flavored (phoneme text encoder →
duration + prosody predictors → decoder + ISTFT vocoder), plus a per-voice
style vector. What torchic is still missing:

### Ops

- [x] `Conv1D` fwd (Workers, WASM, WebGPU)
- [x] `ConvTranspose1D` fwd (Workers, WASM, WebGPU)
- [x] `LayerNorm` — transformer blocks
- [ ] `GroupNorm` / `InstanceNorm` — decoder normalization
- [x] `GELU` — transformer feed-forward
- [x] `sigmoid` — LSTM gates
- [x] `LeakyReLU`, `SiLU` — decoder activations (Workers, WASM, WebGPU)
- [x] `LSTM` cell (PyTorch-layout weight_ih/weight_hh, composed from
      matmul + sigmoid + tanh + slice; bidirectional/sequence wrapper TBD)
- [x] Multi-head attention (composed from Linear + BMM + softmax + transpose;
      a fused GPU kernel is a later perf pass)
- [ ] `Concat` / `Split` on arbitrary axes
- [ ] `Gather` / advanced indexing beyond `embedding`
- [ ] Rotary embeddings (nice-to-have; only if the exact block wants them)
- [ ] `Pad` (reflect + replicate for causal conv stacks)
- [ ] `ISTFT` — vocoder tail (can start as a CPU-side implementation, GPU
      later)

### Loaders

- [ ] Safetensors reader (streaming fetch → parse header → mmap-like
      `Float32Array` views over an `ArrayBuffer`)
- [ ] BF16 → F32 upcast on load (Kokoro weights ship as bf16)
- [ ] Weight-name remapping helper for checkpoints that don't line up 1:1 with
      our `state_dict` keys

### Runtime

- [ ] `Module.forward` in `noGrad` by default in `eval()` — right now nothing
      forces graph-off; a wrapper on top of `eval()` would prevent accidental
      autograd overhead during serving
- [ ] Explicit tensor lifetime API — the graph tracker recycles per
      `trackTensors`; for streaming inference we want to reuse buffers across
      chunks
- [ ] KV/state cache for the LSTM cells (per-utterance)
- [ ] Text frontend: G2P (grapheme → phoneme). Kokoro uses eSpeak-NG's
      phonemes; can either ship a small ONNX G2P or shell out to a JS eSpeak
      port
- [ ] Voice style bank loader (Kokoro voices are separate `.pt` blobs)

### Serving loop

- [ ] AudioWorklet sink so PCM chunks stream to `AudioContext` without gaps
- [ ] Backpressure signal from the sink so we don't overproduce
- [ ] Cancellation token so a new prompt aborts the in-flight decode

### Perf / correctness

- [ ] Fused MHA on WebGPU (matmul + softmax + matmul in one pipeline)
- [ ] BF16 → F32 upcast on the GPU without a CPU round-trip
- [ ] `Conv1D` on WebGPU via im2col + matmul, or a direct kernel
- [ ] Bench harness that reports RTF (real-time factor), not just ms/step

## Milestones

Each milestone should land a demo in `tests/demos/` that the headless bench
harness (see `scripts/bench.mjs`) can run.

1. **M1 — LayerNorm + GELU + a working MLP transformer block.**
   Enough to build a `nn.MultiHeadAttention` and a `nn.TransformerEncoderLayer`
   from primitives. Demo: BERT-tiny-style masked-LM overfit on a tiny corpus.

2. **M2 — Safetensors loader + BF16 upcast.**
   Load any HF `.safetensors` file into a matching `Module` tree.
   Demo: load a small pre-trained model (e.g. `bert-tiny`) and verify a
   forward pass matches HF within tolerance.

3. **M3 — Conv1D + LSTM.**
   Get the two big Kokoro-specific ops on Workers first, then WASM, then
   WebGPU. Demo: WaveNet-lite text-to-audio-envelope overfit.

4. **M4 — ISTFT.**
   Start with a CPU (main-thread) implementation using FFT.
   Demo: mel-spectrogram → PCM round-trip.

5. **M5 — Kokoro forward pass (weights loaded, one voice, unbatched).**
   Not yet performant. Just prove numerics match the reference within
   audible tolerance.

6. **M6 — Streaming synthesis via AudioWorklet.**
   Chunked decode + PCM handoff. Report RTF from the bench harness.

7. **M7 — WebGPU fused kernels.** Attention + Conv1D speedups. RTF < 1.0.

## Open questions

- **Do we ship a G2P?** Kokoro is phoneme-input. Options: (a) run eSpeak-NG
  compiled to WASM, (b) ship an ONNX G2P through a mini onnx runtime, (c)
  require the caller to pass phonemes. (c) is easiest but useless for demos.
- **Where do voice packs live?** ~500KB each. Ship a manifest and fetch on
  demand? Bundle "af_bella" only?
- **BF16 storage in the browser.** Do we upcast eagerly on load (2x memory)
  or keep BF16 and expand per-op? Eager is simpler; keep as F32 in memory,
  document that 82M params ≈ 328 MB.
- **How much of Kokoro's pipeline is safe to keep on main thread?** Text
  frontend + ISTFT can probably stay off the compute backends without hurting
  RTF.

## Non-goals (for now)

- Training. This is a serving library first.
- Multiple concurrent utterances.
- Non-English voices.
- Real-time voice cloning.
