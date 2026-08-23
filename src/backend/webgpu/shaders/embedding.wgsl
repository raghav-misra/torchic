// Embedding lookup + vocab-major backward.
//
// Forward: output[i] = weights[indices[i / D] * D + (i % D)] where D = embedding_dim.
//
// Backward: for each (row, col) in weights_grad, sum output_grad rows whose
// index matched `row`. Vocab-major (one thread per output weight element)
// avoids the scatter-races that block a naive parallel port and keeps us off
// f32 atomics, which WGSL doesn't have.

struct EmbeddingU {
  // Forward: weights_off | Backward: weights_grad_off
  buf_w: u32,
  // Both: indices_off
  buf_i: u32,
  // Forward: output_off | Backward: output_grad_off
  buf_o: u32,
  embedding_dim: u32,
  // Forward: total output elements | Backward: total weights_grad elements
  count: u32,
  // Backward only: number of indices (rows of output_grad)
  num_indices: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: EmbeddingU;

@compute @workgroup_size(256)
fn embedding(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }
  let idx_i = i / u.embedding_dim;
  let off = i % u.embedding_dim;
  let row = u32(heap[u.buf_i + idx_i]);
  heap[u.buf_o + i] = heap[u.buf_w + row * u.embedding_dim + off];
}

@compute @workgroup_size(64)
fn embedding_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }
  let row = i / u.embedding_dim;
  let col = i % u.embedding_dim;
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < u.num_indices; k = k + 1u) {
    let idx = u32(heap[u.buf_i + k]);
    if (idx == row) {
      acc = acc + heap[u.buf_o + k * u.embedding_dim + col];
    }
  }
  // Accumulate to match the workers/wasm scatter kernel; frontend FILLs weights_grad to 0 first.
  heap[u.buf_w + i] = heap[u.buf_w + i] + acc;
}
