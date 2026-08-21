// Row-parallel 2D softmax. One thread per row.
// Shared uniform struct — softmax uses p0/p1, softmax_backward uses p0/p1/p2.

struct SoftmaxU {
  p0: u32,   // softmax: input      | softmax_backward: output
  p1: u32,   // softmax: output     | softmax_backward: grad_output
  p2: u32,   // softmax: (unused)   | softmax_backward: grad_input
  m: u32,
  n: u32,
  start_row: u32,
  end_row: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: SoftmaxU;

@compute @workgroup_size(64)
fn softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = u.start_row + gid.x;
  if (row >= u.end_row) { return; }
  let in_base = u.p0 + row * u.n;
  let out_base = u.p1 + row * u.n;

  var maxv: f32 = -1e38;
  for (var c: u32 = 0u; c < u.n; c = c + 1u) {
    let v = heap[in_base + c];
    if (v > maxv) { maxv = v; }
  }

  var sum: f32 = 0.0;
  for (var c: u32 = 0u; c < u.n; c = c + 1u) {
    let e = exp(heap[in_base + c] - maxv);
    heap[out_base + c] = e;
    sum = sum + e;
  }

  if (sum != 0.0) {
    let inv = 1.0 / sum;
    for (var c: u32 = 0u; c < u.n; c = c + 1u) {
      heap[out_base + c] = heap[out_base + c] * inv;
    }
  } else {
    let v = 1.0 / f32(u.n);
    for (var c: u32 = 0u; c < u.n; c = c + 1u) {
      heap[out_base + c] = v;
    }
  }
}

// grad_input[r, c] = output[r, c] * (grad_output[r, c] - dot(grad_output[r, :], output[r, :]))
@compute @workgroup_size(64)
fn softmax_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = u.start_row + gid.x;
  if (row >= u.end_row) { return; }
  let out_base = u.p0 + row * u.n;
  let go_base = u.p1 + row * u.n;
  let gi_base = u.p2 + row * u.n;

  var dot: f32 = 0.0;
  for (var c: u32 = 0u; c < u.n; c = c + 1u) {
    dot = dot + heap[go_base + c] * heap[out_base + c];
  }
  for (var c: u32 = 0u; c < u.n; c = c + 1u) {
    heap[gi_base + c] = heap[out_base + c] * (heap[go_base + c] - dot);
  }
}
