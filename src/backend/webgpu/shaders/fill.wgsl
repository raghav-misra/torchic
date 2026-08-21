// Fill a range with a scalar value. val_bits is bitcast of f32 → u32 so it
// travels through the uniform buffer as an integer.

struct FillU {
  output_off: u32,
  start: u32,
  end: u32,
  val_bits: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: FillU;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  heap[u.output_off + i] = bitcast<f32>(u.val_bits);
}
