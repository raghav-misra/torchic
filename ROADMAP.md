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
- [x] `nn.BiLSTM` on top of `LSTMCell` (concat over 2*hidden output dim)
- [x] `Tensor.concat(tensors, axis)`, `Tensor.split(sections, axis)`,
      `Tensor.pad1d(left, right, value)` — Concat kernel on all three backends
- [x] Kokoro model skeleton in `src/models/kokoro/`: `AdaIN1d`,
      `AdaLayerNorm`, `AdainResBlk1d`, `TextEncoder`, `DurationEncoder`,
      `ProsodyPredictor`, `ResBlock1`, `MRF`, `ISTFTGenerator`, `PLBERT`,
      `Kokoro`. ~105M params, right order of magnitude; exact name/shape
      alignment with the real state_dict happens at demo review.

## Gap analysis

Kokoro's public architecture is StyleTTS 2 flavored (phoneme text encoder →
duration + prosody predictors → decoder + ISTFT vocoder), plus a per-voice
style vector. What torchic is still missing:

### Ops

- [x] `Conv1D` fwd (Workers, WASM, WebGPU)
- [x] `ConvTranspose1D` fwd (Workers, WASM, WebGPU)
- [x] `LayerNorm` — transformer blocks
- [x] `GroupNorm` / `InstanceNorm1d` (composed from primitives)
- [x] `GELU` — transformer feed-forward
- [x] `sigmoid` — LSTM gates
- [x] `LeakyReLU`, `SiLU` — decoder activations (Workers, WASM, WebGPU)
- [x] `LSTM` cell (PyTorch-layout weight_ih/weight_hh, composed from
      matmul + sigmoid + tanh + slice; bidirectional/sequence wrapper TBD)
- [x] Multi-head attention (composed from Linear + BMM + softmax + transpose;
      a fused GPU kernel is a later perf pass)
- [x] `Concat` on arbitrary axes (Workers, WASM, WebGPU); `Split` zero-copy
- [x] `Pad1d` (constant mode via concat)
- [ ] `Gather` / advanced indexing beyond `embedding`
- [ ] Rotary embeddings (nice-to-have; only if the exact block wants them)
- [ ] `Pad` reflect / replicate modes (only if the model actually needs them
      — constant is enough for a first Kokoro pass)
- [x] `STFT` / `ISTFT` — main-thread CPU DSP (src/dsp/), pow2 FFT, Hann
      window, WOLA reconstruction

### Loaders

- [x] Safetensors reader (header parse + F32/BF16/F16 upcast, in
      src/nn/safetensors.ts)
- [x] BF16 → F32 upcast on load
- [x] Weight-name remapping helper via `Module.load_safetensors(sd, {renameMap})`
- [x] `Module.load_safetensors(map, {strict, renameMap})` — writes
      Float32Array data straight into destination tensors, skips the
      Tensor-alloc round-trip of `load_state_dict`
- [x] `saveSafetensors(map, metadata?)` — writes an ArrayBuffer we can
      round-trip in tests

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

1. **M1 — LayerNorm + GELU + a working MLP transformer block.** ✅ Done —
   `nn.MultiHeadAttention`, `nn.TransformerEncoderLayer`, positional
   encoding, all activations on all three backends.

2. **M2 — Safetensors loader + BF16 upcast.** ✅ Done for the loader,
   writer, and Module integration. `Module.load_safetensors(map,
   {strict, renameMap})` is verified via round-trip on all three
   backends. M2b (numerical parity against a real HF checkpoint) will
   fall out of the Kokoro demo review.

3. **M3 — Conv1D + LSTM.** ✅ Done for Conv1d/ConvTranspose1d (all three
   backends), LSTMCell + BiLSTM. Kokoro-specific extras (grouped conv,
   weight_norm's g/v layout) will be added in the demo-review pass.

4. **M4 — ISTFT.** ✅ Done as main-thread CPU DSP in `src/dsp/`.
   Kokoro's `gen_istft_n_fft=20, hop=5` config is trivial for this
   implementation.

5. **M5 — Kokoro forward pass (weights loaded, one voice, unbatched).**
   ⏸ At the STOP point. The full module tree is in `src/models/kokoro/`
   with param count ~105M (target 82M — the demo review will shave the
   over-count by aligning names/shapes with the real checkpoint's
   state_dict). Not yet wired to real weights per user instruction.

6. **M6 — Streaming synthesis via AudioWorklet.** Chunked decode + PCM
   handoff. Report RTF from the bench harness.

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
