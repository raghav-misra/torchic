// Gather a strided view into a contiguous output.
// Shape/strides packed into vec4<u32> arrays to satisfy WGSL uniform alignment
// (each u32 in a uniform array would otherwise be 16-byte-aligned = wasteful).
// Max supported rank = 8.

struct MaterializeU {
  input_off: u32,
  output_off: u32,
  ndim: u32,
  count: u32,
  shape: array<vec4<u32>, 2>,
  strides: array<vec4<u32>, 2>,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: MaterializeU;

fn shape_at(dim: u32) -> u32 {
  return u.shape[dim / 4u][dim % 4u];
}

fn strides_at(dim: u32) -> u32 {
  return u.strides[dim / 4u][dim % 4u];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }

  var idx = i;
  var input_offset: u32 = 0u;
  var d = u.ndim;
  loop {
    if (d == 0u) { break; }
    d = d - 1u;
    let size = shape_at(d);
    let pos = idx % size;
    idx = idx / size;
    input_offset = input_offset + pos * strides_at(d);
  }
  heap[u.output_off + i] = heap[u.input_off + input_offset];
}
