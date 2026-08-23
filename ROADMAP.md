# torchic → Kokoro TTS: roadmap

We ship an in-browser Kokoro-82M synthesis pipeline that runs end-to-end.
Three parallel tracks from here:

## Track 1: long-form synthesis

Today's demo synthesizes ~2 s at a time before it drifts / cuts off. Get
comfortable with paragraph-length inputs.

- [ ] Diagnose "last word rushes / cuts off" on 40+ token inputs. Log
      `predDur` per phoneme and audio length vs expected to isolate whether
      it's a duration-head drift or a length-computation off-by-one.
- [ ] Sentence splitting: break long text at punctuation, synthesize
      per-sentence, concat PCM with a short silence pad. Sidesteps whatever
      cap the current pipeline has and gives the caller an obvious knob.
- [ ] Ship a real G2P frontend (`misaki` or eSpeak-NG WASM) so the demo
      accepts text, not just pre-tokenized phoneme IDs.
- [ ] Bundle more voice packs in the demo folder.

## Track 2: RTF < 1.0

Today: RTF ~5 on WebGPU for "hello world" (baseline 15-token utterance).
Target: **< 1.0** so audio is produced faster than it plays back.

- [ ] **Profile.** Where is the time actually going? Instrument
      `Kokoro.forward` to log ms per stage (bert, predictor, decode, generator,
      istft). Bench harness reports RTF too.
- [ ] **Fused MHA on WebGPU** (Q·Kᵀ + softmax + ·V in one pipeline). BERT
      is 12 layers × attention — likely the top hotspot.
- [ ] **Reduce dispatch overhead.** Each op is one WebGPU dispatch; deep
      networks with tiny tensors are latency-bound. Investigate batching or
      workgraph-style scheduling.
- [ ] **`Conv1d` on WebGPU via im2col + matmul.** Convs feed into every
      resblock; today they're the direct-loop kernel.
- [ ] **Better WebGPU matmul kernel.** `vec4<f32>` loads, transposed thread
      mapping to fix bank conflicts, 128×128 workgroup tiles. Target 1–2
      TFLOPS on 2070-class GPUs (~3–6× today's 310 GFLOPS).
- [ ] **BF16 in-GPU upcast.** Load safetensors as BF16 into the heap, upcast
      in the shader instead of on the CPU. Halves memory bandwidth and lets
      us keep the checkpoint smaller.

## Track 3: quality parity with reference PyTorch

Today: recognizable but "muffled." Target: indistinguishable from a
`kokoro-python` synthesis of the same phoneme sequence on the same voice.

- [ ] **Numerical parity harness.** Python script that runs reference Kokoro
      on the same phoneme IDs and dumps intermediates at 6–8 checkpoints
      (`bert_out`, `d_en`, `duration_logits`, `en`, `F0_pred`, `N_pred`,
      `t_en`, `asr`, final PCM). JS side dumps the same. Diff L2 distance
      per stage → wherever it jumps sharply is the bug.
- [ ] Restore the `SineGen` `F.interpolate` phase smoothing correctly
      (currently reverted — added end-of-utterance cutoff). Careful boundary
      handling to match PyTorch's `align_corners=False` exactly.
- [ ] Check inline `stftHann` DFT sign/scale/window against `torch.stft` for
      nFFT=20.
- [ ] Verify `BiLSTM` numerical output matches PyTorch's fused-gate `nn.LSTM`
      to within f32 eps.
- [ ] Check softmax / attention numerical stability on WebGPU (hardware
      `exp` precision).

## Non-goals

- Training. Serving library.
- Multiple concurrent utterances.
- Streaming synthesis (chunked decoder state). Full-utterance is the
  current model.
- Non-English voices (would need multilingual G2P).
