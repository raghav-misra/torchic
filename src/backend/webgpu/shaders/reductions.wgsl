// Two-phase sum reduction. sum_partial reduces slices to per-slot partials in
// a temp region; sum_final reduces the partials to a scalar output. sum_axis
// walks (outer, axis, inner) and writes one output per (outer, inner).

struct SumU {
  input_off: u32,
  output_off: u32,
  count: u32,
  num_partials: u32,
  axis_size: u32,
  inner_size: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: SumU;

// One thread per partial slot. Each thread sums its slice sequentially.
// count elements split across num_partials slots.
@compute @workgroup_size(64)
fn sum_partial(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= u.num_partials) { return; }
  let per = (u.count + u.num_partials - 1u) / u.num_partials;
  let start = slot * per;
  var end = start + per;
  if (end > u.count) { end = u.count; }
  var sum: f32 = 0.0;
  for (var i: u32 = start; i < end; i = i + 1u) {
    sum = sum + heap[u.input_off + i];
  }
  heap[u.output_off + slot] = sum;
}

// Single-thread reduction of the small partials array (up to num_partials).
@compute @workgroup_size(1)
fn sum_final(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x > 0u) { return; }
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < u.num_partials; i = i + 1u) {
    sum = sum + heap[u.input_off + i];
  }
  heap[u.output_off] = sum;
}

// One thread per output element. Sums axis_size entries stepping by inner_size.
@compute @workgroup_size(64)
fn sum_axis(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }
  let outer = i / u.inner_size;
  let inner = i % u.inner_size;
  let base = outer * u.axis_size * u.inner_size + inner;
  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < u.axis_size; k = k + 1u) {
    sum = sum + heap[u.input_off + base + k * u.inner_size];
  }
  heap[u.output_off + i] = sum;
}
