// Concat: write one contiguous input into a strided slab of the output.
// One thread per input element; caller dispatches once per input tensor with
// the appropriate axis_offset.

struct ConcatU {
  input_off: u32,
  output_off: u32,
  outer_size: u32,
  in_axis_size: u32,
  out_axis_size: u32,
  axis_offset: u32,
  inner_size: u32,
  total: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: ConcatU;

@compute @workgroup_size(64)
fn concat_slab(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= u.total) { return; }
  let axis_x_inner = u.in_axis_size * u.inner_size;
  let out_axis_x_inner = u.out_axis_size * u.inner_size;
  let outer = idx / axis_x_inner;
  let rem = idx % axis_x_inner;
  let axis = rem / u.inner_size;
  let inner = rem % u.inner_size;
  let out_idx = outer * out_axis_x_inner + (u.axis_offset + axis) * u.inner_size + inner;
  heap[u.output_off + out_idx] = heap[u.input_off + idx];
}
