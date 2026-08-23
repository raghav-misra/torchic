# torchic → Kokoro TTS: model-serving roadmap

## North star

Load the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS checkpoint
in the browser and synthesize an English utterance end-to-end, running
entirely on torchic. Target: RTF < 1.0 on a modern laptop with WebGPU (audio
produced faster than it plays back). Full-utterance synthesis, not streaming.

```ts
import { init } from "torchic";
import { Kokoro } from "./tests/demos/kokoro";

await init({ backend: "webgpu", memorySizeMB: 512 });
const model = await Kokoro.fromPretrained("hexgrad/Kokoro-82M");
const pcm = await model.synthesize(phonemes, { voice: "af_bella" });
const buf = audioCtx.createBuffer(1, pcm.length, 24000);
buf.copyToChannel(pcm, 0);
audioCtx.createBufferSource().buffer = buf;
```

## Current state

All primitives Kokoro needs are implemented across Workers, WASM, and WebGPU,
verified
via headless parity suites (`scripts/bench.mjs`). Kokoro itself lives
as a demo under `tests/demos/kokoro/` — the library ships primitives only.
The module tree instantiates end-to-end at ~104M params (target 82M —
overcount will be resolved during shape alignment against the real state_dict).
Not yet wired to real weights.

## What's left

### Runtime

- [ ] `Module.eval()` forces `noGrad` for the forward pass — no accidental
      autograd overhead during serving

### Kokoro-specific (lives under `tests/demos/kokoro/`)

- [ ] Align skeleton state_dict names + shapes with the real checkpoint
      (closes the ~22M param overshoot)
- [ ] `weight_norm` g/v layout, if the checkpoint stores weights pre-fusion
- [ ] Wire `Kokoro.forward()` — currently throws
- [ ] Bundle one voice pack (e.g. `af_bella`) with the demo
- [ ] Caller passes phonemes for now (no G2P shipped)

### Nice-to-haves

- [ ] `Gather` / advanced indexing beyond `embedding` (needed to expand
      per-phoneme features by predicted durations into the frame axis)
- [ ] `Pad` reflect / replicate modes
- [ ] Rotary embeddings (only if a Kokoro sub-block needs them)
- [ ] Fused MHA on WebGPU (matmul + softmax + matmul in one pipeline)
- [ ] `Conv1D` on WebGPU via im2col + matmul
- [ ] BF16 → F32 upcast on the GPU without a CPU round-trip
- [ ] Bench harness reports RTF = `wall_time / (num_samples / sample_rate)`
