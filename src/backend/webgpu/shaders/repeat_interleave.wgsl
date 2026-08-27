// Repeat-interleave: duplicate each slice along an axis `repeats` times.
// Treats input as [outer, axis_size, inner] and output as [outer, axis_size * repeats, inner].
// For output index k → (o, d_out, i), map back to input index (o, d_out / repeats, i).

struct RepeatInterleaveU {
  input: u32,
  output: u32,
  axis_size: u32,
  inner: u32,
  repeats: u32,
  total: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: RepeatInterleaveU;

@compute @workgroup_size(256)
fn repeat_interleave(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= u.total) { return; }
  let d_out_stride = u.axis_size * u.repeats;
  let inner_idx = k % u.inner;
  let rest = k / u.inner;
  let d_out = rest % d_out_stride;
  let o = rest / d_out_stride;
  let d_in = d_out / u.repeats;
  let input_idx = o * u.axis_size * u.inner + d_in * u.inner + inner_idx;
  heap[u.output + k] = heap[u.input + input_idx];
}
