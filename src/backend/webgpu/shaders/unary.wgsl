// Unary elementwise ops: 2 pointers + start/end range.

struct UnaryU {
  input_off: u32,
  output_off: u32,
  start: u32,
  end: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: UnaryU;

@compute @workgroup_size(256)
fn neg(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  heap[u.output_off + i] = -heap[u.input_off + i];
}

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  heap[u.output_off + i] = max(0.0, heap[u.input_off + i]);
}

@compute @workgroup_size(256)
fn exp_(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  heap[u.output_off + i] = exp(heap[u.input_off + i]);
}

@compute @workgroup_size(256)
fn log_(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  heap[u.output_off + i] = log(heap[u.input_off + i]);
}

@compute @workgroup_size(256)
fn tanh_(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  heap[u.output_off + i] = tanh(heap[u.input_off + i]);
}

@compute @workgroup_size(256)
fn copy_(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  heap[u.output_off + i] = heap[u.input_off + i];
}
