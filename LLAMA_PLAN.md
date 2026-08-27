# Llama 3.2 on torchic — Plan

Serve Llama 3.2 (1B for correctness, 3B for real) end-to-end in-browser on top of the torchic backends.

## Code layout

Library (`src/`) stays model-agnostic: tensor primitives, backends, autograd, `nn/` layers, and reusable ops any decoder-only LLM would want. Model architectures live in `tests/demos/` (same as `tests/demos/kokoro/`). Llama-specific code goes in `tests/demos/llama/`.

## Gap check

- [x] RoPE (rotary position embeddings)
- [x] Causal-masked softmax
- [x] Slice-write primitive (`Tensor.copyFrom(source, { atOffset })`) — foundation for KV cache
- [ ] Repeat-interleave op — for GQA broadcasting
- [ ] `functional.causalAttention(q, k, v, pastLen)` — composed helper
- [ ] `nn.KVCache` — thin per-layer bookkeeping class
- [ ] BPE tokenizer + chat template
- [ ] Autoregressive sampling loop
- [ ] INT4 loader + dequant path

## Design gotchas to keep in mind

- [ ] Prefill (matmul) and decode (matvec) are different kernels — decode is memory-bandwidth-bound, needs its own dispatch shape.
- [ ] KV cache dominates memory: ~112 KB per token at F16 across all layers. Cap `max_seq_len` (2K–8K), pre-allocate, append via slice-writes not concat.
- [ ] Embedding table is `128256 × 3072` — ~750 MB at F16, exceeds WebGPU 256 MB guaranteed binding size. CPU-side lookup during decode, or shard, or wait for INT4.
- [ ] Full F16 3B (~6 GB) won't fit on most consumer GPUs. Do correctness on **Llama 3.2 1B**, scale to 3B once INT4 works.

## Phase 1 — new primitives (inference-only, no backward)

- [x] `rms_norm(x, weight, eps)` kernel across all three backends
- [x] `RMSNorm` layer + parity tests (workers ↔ wasm ↔ webgpu, tol 1e-5)
- [x] `rope(x, cos, sin)` — half-split HF convention, `[..., T, D]` input, `[T, D/2]` caches
- [x] RoPE kernel across all three backends
- [x] `precomputeRope(maxSeqLen, headDim, theta)` helper
- [x] `causal_softmax(pastLen)` fused kernel across all three backends
- [x] `SwiGLU` FFN layer in `nn/` (composition of Linear + silu + mul, no new kernel)

## Phase 2 — library primitives for causal decoding

Everything here goes in `src/` — generic across any decoder-only LLM (Llama, Mistral, Qwen, Phi3, ...). No Llama-specific assumptions.

### 2a. Slice-write primitive
- [x] `Tensor.copyFrom(source, { atOffset })` semantics: overwrite this tensor's storage starting at `atOffset` elements with the contiguous data from `source`. Shape/stride agnostic on `source`; only requires enough space in `this`.
- [x] `COPY_RANGE` op wired through the three backends (workers, wasm, webgpu) — each is a straight `memcpy`/`v128` copy, no reduction.
- [x] Unit test + parity test.

### 2b. Repeat-interleave (for GQA)
- [ ] `Tensor.repeatInterleave(dim, count)` — e.g. `[B, 8, T, D].repeatInterleave(1, 3) → [B, 24, T, D]` where each of the 8 heads is duplicated 3× consecutively along dim 1.
- [ ] `REPEAT_INTERLEAVE` op across three backends.
- [ ] Unit test + parity test.

### 2c. Causal attention composition helper
- [ ] `functional.causalAttention(q, k, v, pastLen = 0)` — `q [B, H, T_q, D]`, `k/v [B, H, T_k, D]` where `T_k = pastLen + T_q`. Composes: `bmm(q, k.transpose(-2, -1)) * (1/sqrt(D)) → causal_softmax(pastLen) → bmm(attn, v)`. No new kernel.
- [ ] Unit test (small deterministic input, compare against manual reference).

### 2d. KV cache bookkeeping
- [ ] `nn.KVCache(numLayers, maxSeqLen, numKvHeads, headDim)` — thin wrapper around `2 * numLayers` pre-allocated tensors of shape `[maxSeqLen, numKvHeads, headDim]` plus an int cursor.
- [ ] `.write(layerIdx, kNew, vNew)` — uses `copyFrom` at the current cursor; increments cursor exactly once after ALL layers for the current step have been written.
- [ ] `.read(layerIdx)` — returns `{ k, v }` slice views up to cursor.
- [ ] `.reset()` — cursor := 0.
- [ ] `.position` getter.
- [ ] Unit test simulating multi-step append + read pattern.

## Phase 3 — Llama demo in `tests/demos/llama/`

Everything here is Llama-specific and lives outside `src/`.

- [ ] `config.ts` — `LlamaConfig` (hidden_size, n_layers, n_heads, n_kv_heads, head_dim, ffn_dim, rope_theta, vocab_size, rms_eps, tied_embeddings).
- [ ] `attention.ts` — `LlamaAttention` Module: Q/K/V/O projections, applies RoPE, calls `functional.causalAttention`, manages its layer index into a shared `KVCache`. GQA broadcasting via `.repeatInterleave` for the first cut.
- [ ] `mlp.ts` — SwiGLU FFN wired to Llama's dim ratios.
- [ ] `decoder.ts` — `LlamaDecoderLayer` (RMSNorm → attn → residual → RMSNorm → SwiGLU → residual), `LlamaModel` (embedding + N layers + final RMSNorm).
- [ ] `model.ts` — `LlamaForCausalLM` with tied embedding as LM head.
- [ ] `weights.ts` — HF safetensors key → torchic module tree mapping, weight loader.
- [ ] Load Llama 3.2 1B weights, verify shapes.
- [ ] `parity.suite.ts` — parity harness against `transformers.js` on the same prompt.
  - [ ] Hidden-state parity per layer (cosine similarity ≥ ~0.999)
  - [ ] Logit parity on final output (~1e-3)

## Phase 4 — tokenizer + generation loop (still in demo)

- [ ] Integrate a BPE tokenizer (don't roll your own — use `@huggingface/tokenizers` or port from transformers.js).
- [ ] Apply Llama 3.2 chat template (`<|begin_of_text|><|start_header_id|>...`).
- [ ] Greedy decode (argmax) — verify token stream matches HF byte-for-byte.
- [ ] Temperature / top-k / top-p sampling (library helpers in `nn/functional`, sampling is generic).
- [ ] Streaming API: `AsyncIterable<string>` yielding decoded token text.
- [ ] Stop-token handling (`<|eot_id|>`, `<|end_of_text|>`).

## Phase 5 — quantization

- [ ] Pick a format: AWQ safetensors (simpler) or GGUF Q4_K_M (more common).
- [ ] Quantized weight loader (packed int words + FP16 scales + zero-points) — extension to `src/nn/safetensors.ts`.
- [ ] Dequant-then-matmul path: unpack to F16 scratch buffer, reuse existing matmul.
- [ ] Parity vs F16 (~1% logit divergence expected).
- [ ] Fused dequant+matmul WGSL kernel — unpack inside inner loop.
- [ ] Fused dequant+matvec WGSL kernel — the one that decides tok/s.
- [ ] Scale to Llama 3.2 3B once INT4 correctness holds.

## Phase 6 — matvec + serving polish

- [ ] Dedicated matvec kernel (one workgroup per output row, coalesced weight reads) — library op.
- [ ] Multi-turn: cache reset on new conversation, retained on continuation — demo.
- [ ] Cancel/abort mid-generation — demo.
- [ ] tok/s metering + generation progress events — demo.
- [ ] (Stretch) Speculative decoding: 1B draft model → 3B verifier.

## Immediate next move

Phase 1 finish: `SwiGLU` FFN layer (~10 lines, no new kernel). Then Phase 2a: `Tensor.copyFrom` slice-write primitive across all three backends — the foundation for the KV cache.
