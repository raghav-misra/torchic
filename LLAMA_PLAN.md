# Llama 3.2 on torchic — Plan

Serve Llama 3.2 (1B for correctness, 3B for real) end-to-end in-browser on top of the torchic backends.

## Gap check

- [x] RoPE (rotary position embeddings)
- [x] Causal-masked softmax
- [ ] GQA-aware attention
- [ ] KV cache
- [ ] BPE tokenizer + chat template
- [ ] Autoregressive sampling loop
- [ ] INT4 loader + dequant path

## Design gotchas to keep in mind

- [ ] Prefill (matmul) and decode (matvec) are different kernels — decode is memory-bandwidth-bound, needs its own dispatch shape.
- [ ] KV cache dominates memory: ~112 KB per token at F16 across all layers. Cap `max_seq_len` (2K–8K), pre-allocate, append via slice-writes not concat.
- [ ] Embedding table is `128256 × 3072` — ~750 MB at F16, exceeds WebGPU 256 MB guaranteed binding size. CPU-side lookup during decode, or shard, or wait for INT4.
- [ ] Full F16 3B (~6 GB) won't fit on most consumer GPUs. Do correctness on **Llama 3.2 1B**, scale to 3B once INT4 works.

## Phase 1 — new primitives (inference-only, no backward)

- [x] `rms_norm(x, weight, eps)` kernel — JS worker backend
- [x] `rms_norm` kernel — WASM (Rust SIMD)
- [x] `rms_norm` kernel — WebGPU (WGSL)
- [x] `RMSNorm` layer + parity tests (workers ↔ wasm ↔ webgpu, tol 1e-5)
- [x] `rope_apply(x, cos, sin)` — half-split HF convention, `[..., T, D]` input, `[T, D/2]` caches
- [x] RoPE kernel across all three backends
- [x] `precomputeRope(maxSeqLen, headDim, theta)` helper + parity tests
- [x] Causal-masked softmax (fused kernel, `pastLen` param for prefill continuation)
- [ ] SwiGLU FFN layer (no new kernel — `silu(gate(x)) * up(x) -> down`)
- [ ] Parity tests for each new op against a reference (numpy or transformers.js)## Phase 2 — causal GQA attention with KV cache

- [ ] Design cache layout per layer: `K_cache/V_cache: [max_seq_len, num_kv_heads, head_dim]`, seq cursor
- [ ] Prefill path: batched Q/K/V for N tokens, RoPE over positions `[0..N)`, write to cache `[0..N)`, standard causal attention
- [ ] Decode path: single-token Q, matvec `Q @ K_cache[0:t+1].T`, no mask needed
- [ ] GQA broadcasting: repeat KV heads 3× (simple, correct) as the first cut
- [ ] Unfused attention pipeline: Q@K → mask → softmax → @V — do NOT fuse yet
- [ ] Optimization pass: GQA index-aware attention (skip the KV repeat)
- [ ] Optimization pass: FlashAttention-style fusion (later, only if needed)

## Phase 3 — model assembly + parity

- [ ] `LlamaDecoderLayer` Module (RMSNorm → attn → residual → RMSNorm → SwiGLU → residual)
- [ ] `LlamaModel` — embedding + N decoder layers + final RMSNorm
- [ ] `LlamaForCausalLM` with tied embedding as LM head
- [ ] Weight name mapping table: HF safetensors keys → torchic module tree
- [ ] Load Llama 3.2 1B weights, verify shapes
- [ ] Parity harness against `transformers.js` on the same prompt
- [ ] Hidden-state parity per layer (cosine similarity ≥ ~0.999)
- [ ] Logit parity on final output (~1e-3)

## Phase 4 — tokenizer + generation loop

- [ ] Integrate a BPE tokenizer (don't roll your own — use `@huggingface/tokenizers` or port from transformers.js)
- [ ] Apply Llama 3.2 chat template (`<|begin_of_text|><|start_header_id|>...`)
- [ ] Greedy decode (argmax) — verify token stream matches HF byte-for-byte
- [ ] Temperature sampling
- [ ] Top-k sampling
- [ ] Top-p (nucleus) sampling
- [ ] Streaming API: `AsyncIterable<string>` yielding decoded token text
- [ ] Stop-token handling (`<|eot_id|>`, `<|end_of_text|>`)

## Phase 5 — quantization

- [ ] Pick a format: AWQ safetensors (simpler) or GGUF Q4_K_M (more common)
- [ ] Quantized weight loader (packed int words + FP16 scales + zero-points)
- [ ] Dequant-then-matmul path: unpack to F16 scratch buffer, reuse existing matmul
- [ ] Parity vs F16 (~1% logit divergence expected, acceptable)
- [ ] Fused dequant+matmul WGSL kernel — unpack inside inner loop, no scratch buffer
- [ ] Fused dequant+matvec WGSL kernel — the kernel that decides your tok/s
- [ ] Scale to Llama 3.2 3B once INT4 correctness holds

## Phase 6 — matvec + serving polish

- [ ] Dedicated matvec kernel (one workgroup per output row, coalesced weight reads)
- [ ] Multi-turn: cache reset on new conversation, retained on continuation
- [ ] Cancel/abort mid-generation
- [ ] tok/s metering + generation progress events
- [ ] (Stretch) Speculative decoding: 1B draft model → 3B verifier

## Immediate first move

- [ ] Phase 1 primitives (RMSNorm + RoPE + causal-masked softmax) on **Llama 3.2 1B**, with parity tests, across all three backends.
