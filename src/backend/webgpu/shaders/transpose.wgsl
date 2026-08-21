// Transpose input[m, n] -> output[n, m].
// Each thread writes one output cell.

struct TransposeU {
  input_off: u32,
  output_off: u32,
  m: u32,
  n: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: TransposeU;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;  // row in output (0..n)
  let c = gid.y;  // col in output (0..m)
  if (r >= u.n || c >= u.m) { return; }
  heap[u.output_off + r * u.m + c] = heap[u.input_off + c * u.n + r];
}
