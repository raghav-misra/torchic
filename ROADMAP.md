# torchic → Kokoro TTS: model-serving roadmap

## North star

Load the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS checkpoint
in the browser and stream synthesized audio for arbitrary English text,
running entirely on torchic. Target: RTF < 1.0 on a modern laptop with WebGPU,
real-time audio out via AudioWorklet.

```ts
import { init } from "torchic";
import { Kokoro } from "./tests/demos/kokoro";

await init({ backend: "webgpu", memorySizeMB: 512 });
const model = await Kokoro.fromPretrained("hexgrad/Kokoro-82M");
const audio = await model.synthesize("Hello, world.", { voice: "af_bella" });
audio.play();
```

## Current state

All primitives Kokoro needs are implemented across Workers, WASM, and WebGPU,
verified via headless parity suites (`scripts/bench.mjs`). Kokoro itself lives
as a demo under `tests/demos/kokoro/` — the library ships primitives only.
The module tree instantiates end-to-end at ~104M params (target 82M —
overcount will be resolved during shape alignment against the real state_dict).
Not yet wired to real weights.

## What's left

### Runtime

- [ ] `Module.eval()` forces `noGrad` for the forward pass — no accidental
      autograd overhead during serving
- [ ] Explicit tensor lifetime API for streaming inference (reuse buffers
      across chunks)
- [ ] KV/state cache for LSTM cells (per-utterance)

### Serving loop

- [ ] AudioWorklet sink so PCM chunks stream to `AudioContext` without gaps
- [ ] Backpressure signal from the sink so we don't overproduce
- [ ] Cancellation token so a new prompt aborts the in-flight decode

### Kokoro-specific (lives under `tests/demos/kokoro/`)

- [ ] Align skeleton state_dict names + shapes with the real checkpoint
      (closes the ~22M param overshoot)
- [ ] `weight_norm` g/v layout, if the checkpoint stores weights pre-fusion
- [ ] Wire `Kokoro.forward()` — currently throws
- [ ] Bundle one voice pack (e.g. `af_bella`) with the demo
- [ ] Caller passes phonemes for now (no G2P shipped)

### Nice-to-haves

- [ ] `Gather` / advanced indexing beyond `embedding`
- [ ] `Pad` reflect / replicate modes
- [ ] Rotary embeddings (only if a Kokoro sub-block needs them)
- [ ] Fused MHA on WebGPU (matmul + softmax + matmul in one pipeline)
- [ ] `Conv1D` on WebGPU via im2col + matmul
- [ ] BF16 → F32 upcast on the GPU without a CPU round-trip
- [ ] Bench harness reports RTF (real-time factor), not just ms/step

## Open questions

- **Streaming vs. batch synthesis.** First cut can produce a full utterance
  and hand it to `AudioContext.decodeAudioData`. Streaming via AudioWorklet
  is M6 — do we need chunked decoder state, or can Kokoro's ISTFT tail be
  buffered per phoneme window?
- **RTF measurement.** Compare to what — wall-clock or audio duration?
  Standard is `wall_time / audio_duration_seconds`; needs the bench harness
  to know audio length in samples.