// Row-parallel causal-masked softmax over an m x n matrix, numerically stable.
// Row r attends only to columns [0, past_len + r] (clamped to n-1). Columns
// beyond that are set to 0. past_len supports prefill continuation on top of
// an existing KV cache (past_len=0 for from-scratch prefill).

struct CausalSoftmaxU {
  input: u32,
  output: u32,
  m: u32,
  n: u32,
  past_len: u32,
  t_query: u32,
  start_row: u32,
  end_row: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: CausalSoftmaxU;

@compute @workgroup_size(64)
fn causal_softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = u.start_row + gid.x;
  if (row >= u.end_row) { return; }
  let in_base = u.input + row * u.n;
  let out_base = u.output + row * u.n;

  let q_pos = row % u.t_query;
  var allowed: u32 = u.past_len + q_pos;
  if (allowed >= u.n) { allowed = u.n - 1u; }
  let end_col: u32 = allowed + 1u;

  var maxv: f32 = -1e38;
  for (var c: u32 = 0u; c < end_col; c = c + 1u) {
    let v = heap[in_base + c];
    if (v > maxv) { maxv = v; }
  }

  var sum: f32 = 0.0;
  for (var c: u32 = 0u; c < end_col; c = c + 1u) {
    let e = exp(heap[in_base + c] - maxv);
    heap[out_base + c] = e;
    sum = sum + e;
  }

  if (sum != 0.0) {
    let inv = 1.0 / sum;
    for (var c: u32 = 0u; c < end_col; c = c + 1u) {
      heap[out_base + c] = heap[out_base + c] * inv;
    }
  } else {
    let v = 1.0 / f32(end_col);
    for (var c: u32 = 0u; c < end_col; c = c + 1u) {
      heap[out_base + c] = v;
    }
  }
  for (var c: u32 = end_col; c < u.n; c = c + 1u) {
    heap[out_base + c] = 0.0;
  }
}
